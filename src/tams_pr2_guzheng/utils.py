import rospy

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.gaussian_process as gp
import tf2_ros
import tf2_geometry_msgs
import tams_pr2_guzheng.utils as utils

from geometry_msgs.msg import PoseStamped, PointStamped, Point, Vector3, Pose, Quaternion
from matplotlib import pyplot as plt
from matplotlib import cm
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from typing import Tuple

from tams_pr2_guzheng.msg import RunEpisodeRequest, RunEpisodeGoal
from tams_pr2_guzheng.paths import RuckigPath

def note_to_string(note):
    return note.replace("♯", "is").lower()

def string_to_note(string):
    return string.replace("is", "♯").upper()

def run_params(run_episode, params, finger='ff'):
    req= RunEpisodeRequest(parameters= params.action_parameters, string= params.string, finger= 'ff')
    run_episode.send_goal(RunEpisodeGoal(req))
    run_episode.wait_for_result()
    return run_episode.get_result()

def row_from_result(result):
    row = RuckigPath.from_action_parameters(result.parameters).params_map
    row['finger'] = result.finger

    expected_onsets = [o for o in result.onsets if o.note == utils.string_to_note(row['string'])]
    unexpected_onsets = [o for o in result.onsets if o.note != utils.string_to_note(row['string'])]

    row['onset_cnt'] = len(result.onsets)
    row['unexpected_onsets'] = len(unexpected_onsets)

    row['onsets'] = result.onsets[:]

    row['neighborhood_context'] = 0.0
    if len(expected_onsets) > 0:
        loudest = max(expected_onsets, key= lambda o: o.loudness)
        row['loudness'] = loudest.loudness
        row['detected_note'] = loudest.note
        row['neighborhood_context'] = loudest.neighborhood_context
    elif len(unexpected_onsets) > 0:
        loudest = max(unexpected_onsets, key= lambda o: o.loudness)
        row['loudness'] = loudest.loudness
        row['detected_note'] = loudest.note
    else:
        row['loudness'] = None
        row['detected_note'] = None

    return row

def string_length(string, tf):
    try:
        pt= PointStamped()
        pt.header.frame_id = f"guzheng/{string}/bridge"
        return tf.transform(pt, f"guzheng/{string}/head").point.x
    except tf2_ros.TransformException as e:
        raise Exception(f"Do not know string length of target string {string}: {str(e)}")

def stitch_paths(paths, tf, frame= 'base_footprint'):
    stitched = Path()
    stitched.header.frame_id = frame

    try:
        for path in paths:
                for pose in path.poses:
                    p = PoseStamped()
                    p.header.frame_id = path.header.frame_id
                    p.pose = pose.pose
                    stitched.poses.append(
                        tf.transform(p, stitched.header.frame_id)
                        )
    except tf2_ros.TransformException as e:
        rospy.logerr(f"could not transform path in frame '{path.header.frame_id}' to '{frame}' while stitching")
        raise e

    return stitched

_image_publisher = {}
_cv_bridge = None
def publish_figure(topic_name, fig):
    from matplotlib import pyplot as plt

    global _cv_bridge
    if _cv_bridge is None:
        import cv_bridge
        _cv_bridge = cv_bridge.CvBridge()

    if topic_name not in _image_publisher:
        _image_publisher[topic_name] = rospy.Publisher(f"~{topic_name}", Image, queue_size=1, latch=True)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    img = img[:,:, [2, 1, 0]] # RGB -> BGR
    plt.close(fig)
    _image_publisher[topic_name].publish(_cv_bridge.cv2_to_imgmsg(img, encoding="bgr8"))

_say_pub = None
def say(txt):
    global _say_pub
    from std_msgs.msg import String as StringMsg
    _say_pub = rospy.Publisher("/say", StringMsg, queue_size= 1, tcp_nodelay= True, latch= True)
    _say_pub.publish(txt)

def normalize(x, params = None):
    if params is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        params = (mean, std)
        return (x - mean) / std, params
    else:
        (mean, std) = params
        return (x - mean) / std

def undo_normalize(x, params):
    (mean, std) = params
    return x * std + mean

def score_safety(df):
    # minimum distance to neighbors to consider safe
    # empirically determined accuracy of string fitting
    # safe_threshold = 0.001 # m

    # distance to saturation of distance safety score
    # saturation_threshold  = 0.015 # m

    # loudness cut-off
    loudness_threshold = 65.0 # dBA

    # a = 1/(saturation_threshold-safe_threshold)
    # b = -a*safe_threshold

    # create new pandas series with same index as df
    scores = pd.Series(np.full(len(df), 0.5), index= df.index, name='safety')

    #scores = (a*df['min_distance']+b)
    #scores[df['min_distance'] >= saturation_threshold] = 1.0
    scores[df['loudness'].isna()] = -0.5
    scores[df['loudness'] > loudness_threshold] = -0.5
    scores[df['unexpected_onsets'] > 0] = -0.5

    if 'neighborhood_context' in df.columns:
        scores[df['neighborhood_context'] < 100] = -0.5

    return scores

def fit_gp(features, value, alpha, rbf_length= None, normalize= False, train= False):
    '''
    @param features: (n_samples, n_features)
    @param value: (n_samples,)
    @param alpha: observation noise level (std)
    @param rbf_length: length scale of RBF kernel

    @return: fitted GaussianProcessRegressor
    '''

    kernel = gp.kernels.ConstantKernel(1.0, constant_value_bounds="fixed")
    if train:
        kernel*= gp.kernels.RBF(length_scale= rbf_length)
    elif rbf_length is not None:
        kernel*= gp.kernels.RBF(length_scale= rbf_length, length_scale_bounds="fixed")

    GPR= gp.GaussianProcessRegressor(
        n_restarts_optimizer=100,
        alpha=alpha**2,
        kernel= kernel,
        normalize_y= normalize,
        )
    GPR.fit(
        features.values if hasattr(features, 'values') else features,
        value.values if hasattr(value, 'values') else value
    )
    return GPR

def prob_gt_zero(distributions : Tuple[np.ndarray, np.ndarray]):
    '''
    @param distributions: stack of (mean, std) of a normal distribution with shape (n_samples, 2)

    @return: p(x >= 0;N(mu,std)) for each sample (n_samples,)
    '''
    assert(len(distributions) == 2)
    assert(distributions[0].size == distributions[1].size)

    return 1-stats.norm.cdf(0.0, *distributions)

def make_grid_points(actionspace, grid_size):
    '''
    generate a set of grid points for plotting parameters in 2D
    '''
    xi, yi = np.meshgrid(
        np.linspace(*actionspace.string_position, grid_size),
        np.linspace(*actionspace.keypoint_pos_y, grid_size)
        )
    return pd.DataFrame({ 'string_position': xi.ravel(), 'keypoint_pos_y': yi.ravel() })

def grid_plot(values, actionspace, cmap, ax = None):
    '''
    plot function matching @make_grid_points above
    '''
    if ax is None:
        ax = plt.gca()
    values= values.ravel().reshape( (int(np.sqrt(values.size)),)*2 )
    im= ax.imshow(values, origin='lower', cmap=cmap, extent=(*actionspace.string_position, *actionspace.keypoint_pos_y), aspect='auto')
    plt.colorbar(im)
    ax.grid(False)
    ax.set_xlabel("string position")
    ax.set_ylabel("keypoint pos y")

def nanbar(sm : cm.ScalarMappable, ax : plt.Axes, *, nan, label= 'NaN'):
    '''
    plot a colorbar and a standalone entry below for a special value (e.g. NaN)

    @param sm: ScalarMappable used for the colorbar
    @param ax: Axes to plot the colorbar on
    @param nan: color for the special value, e.g., cmap.get_bad()
    @param label: label for the special value (default 'NaN')
    '''

    cbar = ax.figure.colorbar(sm, ax= ax)
    sm = cm.ScalarMappable(cmap= mpl.colors.ListedColormap([nan]))

    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    divider = make_axes_locatable(cbar.ax)
    nan_ax = divider.append_axes("bottom", size= "5%", pad= "3%", aspect= 1, anchor= cbar.ax.get_anchor())
    nan_ax.grid(visible=False, which='both', axis='both')  # required for Colorbar constructor below
    nan_cbar = mpl.colorbar.Colorbar(ax=nan_ax, mappable=sm, orientation='vertical')
    nan_cbar.set_ticks([0.5], labels=[label])
    nan_cbar.ax.tick_params(length= 0)

def plot_trials(df : pd.DataFrame, col : pd.Series, cmap = None, nan= 'green', nan_label : str = 'miss', ax : plt.Axes= None, norm= None, actionspace : RuckigPath.ActionSpace = None):
    ax = plt.gca() if ax is None else ax
    cmap= sns.cubehelix_palette(as_cmap=True) if cmap is None else cmap
    cmap.set_under(nan) # TODO: shouldn't modify given cmap
    norm = plt.Normalize(col.min(), col.max()) if norm is None else norm

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    art = sns.scatterplot(x='string_position', y='keypoint_pos_y', data=df, hue=col.fillna(norm.vmin-1), hue_norm=norm, palette=cmap, legend=False, ax= ax)
    art.set_title(col.name)

    if actionspace is not None:
        ax.set_xlim(*actionspace.string_position)
        ax.set_ylim(*actionspace.keypoint_pos_y)

    if col.hasnans:
        nanbar(sm, art, nan= cmap.get_under(), label=nan_label)
    else:
        art.figure.colorbar(sm, ax=art)

def plot_mean_std(x, mean, std, ax= None):
    if ax is None:
        ax = plt.gca()
    sns.lineplot(x=x, y= mean, estimator= None, ax= ax)
    ax.fill_between(x, mean-std, mean+std, alpha= 0.5)
    ax.fill_between(x, mean-1.96*std, mean+1.96*std, alpha= 0.25)