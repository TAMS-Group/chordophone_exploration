import rospy

import numpy as np
import tf2_ros
import tf2_geometry_msgs
import tams_pr2_guzheng.utils as utils

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Vector3, Pose, Quaternion
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
    req= RunEpisodeRequest(parameters= params.action_parameters, string= params.string, finger= 'ff')
    run_episode.send_goal(RunEpisodeGoal(req))
    run_episode.wait_for_result()
    return run_episode.get_result()

def row_from_result(result):
    from .paths import RuckigPath
    row = RuckigPath.from_action_parameters(result.parameters).params_map
    row['finger'] = result.finger

    expected_onsets = [o for o in result.onsets if o.note == utils.string_to_note(row['string'])]
    unexpected_onsets = [o for o in result.onsets if o.note != utils.string_to_note(row['string'])]

    row['onset_cnt'] = len(result.onsets)
    row['unexpected_onsets'] = len(unexpected_onsets)

    row['onsets'] = result.onsets[:]

    if len(expected_onsets) > 0:
        row['loudness'] = max([o.loudness for o in expected_onsets])
        row['detected_note'] = result.onsets[-1].note
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