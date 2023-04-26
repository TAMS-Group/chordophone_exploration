import rospy

import tf2_ros
import tf2_geometry_msgs

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Vector3, Pose, Quaternion
from std_msgs.msg import String, Header, ColorRGBA
from visualization_msgs.msg import Marker

from tams_pr2_guzheng.msg import (
    RunEpisodeRequest,
    RunEpisodeGoal
)
from .paths import RuckigPath

def run_params(run_episode, params, finger='ff'):
    req= RunEpisodeRequest(parameters= params.action_parameters, string= params.note, finger= 'ff')
    run_episode.send_goal(RunEpisodeGoal(req))
    run_episode.wait_for_result()
    return run_episode.get_result()

def row_from_result(result):
    row = RuckigPath.from_action_parameters(result.parameters).params_map
    row['onset_cnt'] = len(result.onsets)
    if len(result.onsets) > 0:
        row['loudness'] = result.onsets[-1].loudness
        row['detected_note'] = result.onsets[-1].note
    else:
        row['loudness'] = None
        row['detected_note'] = None
    return row