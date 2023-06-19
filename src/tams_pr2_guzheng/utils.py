import rospy

import numpy as np
import tf2_ros
import tf2_geometry_msgs

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Vector3, Pose, Quaternion
from std_msgs.msg import String, Header, ColorRGBA
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image

from tams_pr2_guzheng.msg import (
    RunEpisodeRequest,
    RunEpisodeGoal
)

def note_to_string(note):
    return note.replace("♯", "is").lower()

def string_to_note(string):
    return string.replace("is", "♯").upper()

def run_params(run_episode, params, finger='ff'):
    req= RunEpisodeRequest(parameters= params.action_parameters, string= params.note, finger= 'ff')
    run_episode.send_goal(RunEpisodeGoal(req))
    run_episode.wait_for_result()
    return run_episode.get_result()

def row_from_result(result):
    from .paths import RuckigPath
    row = RuckigPath.from_action_parameters(result.parameters).params_map
    row['onset_cnt'] = len(result.onsets)
    row['onsets'] = str(result.onsets)
    row['finger'] = result.finger
    if len(result.onsets) > 0:
        row['loudness'] = result.onsets[-1].loudness
        row['detected_note'] = result.onsets[-1].note
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
        raise Exception(f"No string position defined and could not find length of target string {string}: {str(e)}")

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
    plt.close(fig)
    _image_publisher[topic_name].publish(_cv_bridge.cv2_to_imgmsg(img, encoding="bgr8"))