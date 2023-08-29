#!/usr/bin/env python

import librosa
import matplotlib.pyplot as plt; plt.switch_backend('agg')
import numpy as np
import random
import rospkg
import rospy
import scipy.stats as stats
import tams_pr2_guzheng.utils as utils
import tf2_geometry_msgs
import tf2_ros

from actionlib import SimpleActionClient
from math import tau
from tams_pr2_guzheng.paths import RuckigPath
from tams_pr2_guzheng.onset_to_path import OnsetToPath
from tams_pr2_guzheng.utils import string_length, publish_figure, say
from tams_pr2_guzheng.msg import (
    RunEpisodeAction,
    RunEpisodeGoal,
    RunEpisodeRequest)
from visualization_msgs.msg import MarkerArray

known_strings = []
def known_strings_cb(strings):
    '''collect names of all fitted instrument strings'''
    global known_strings
    known_strings = [s.ns for s in strings.markers if len(s.ns) > 0 and ' ' not in s.ns]

def main():
    rospy.init_node('explore')

    # read parameters
    string = rospy.get_param("~string") # string to explore, might be multiple separated by spaces or "all" for all fitted ones
    finger = rospy.get_param("~finger") # finger to use to pluck, one of "ff", "mf", "rf", "th"
    direction = rospy.get_param("~direction") # direction to pluck in (>0 towards the robot, <0 away from the robot), 0.0 for random
    string_position = rospy.get_param("~string_position") # position on the string to pluck, <0 for random
    keypoint_pos = rospy.get_param("~keypoint_pos") # y position of the key point
    storage_path = rospy.get_param("~storage", rospkg.RosPack().get_path("tams_pr2_guzheng") + "/data/plucks_explore.json")

    # validate parameters
    valid_fingers = ("ff", "mf", "rf", "th")
    if finger not in valid_fingers:
        rospy.logfatal(f"invalid finger '{finger}', use one of {valid_fingers}")
        return

    # prepare exploration

    o2p= OnsetToPath(storage= storage_path) # throws on invalid path

    tf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf)

    known_strings_sub = rospy.Subscriber(
        'guzheng/fitted_strings',
        MarkerArray,
        known_strings_cb,
        queue_size=1)

    run_episode = SimpleActionClient("run_episode", RunEpisodeAction)
    run_episode.wait_for_server()

    string_len = 0.0
    try:
        string_len = string_length(string, tf)
    except Exception as e:
        rospy.logerr(e)
        return

    if string_position > string_len:
        rospy.logerr(f"string position {string_position} exceeds known string length {string_len}")
        return

    path = RuckigPath.prototype(
        string = string,
        direction= direction,
    )
    path.string_position = string_position
    path.keypoint_pos[0] = keypoint_pos

    rospy.loginfo(f"Pluck:\n{str(path)}")
    run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= path.action_parameters, string= path.string, finger= finger)))
    run_episode.wait_for_result()

    result = run_episode.get_result()

    # compute minimum distance to other strings during execution (used in safety score)
    minimum_distance = np.inf
    known_strings_cp = known_strings[:]
    for string in known_strings_cp:
        string_frame = f"guzheng/{string}/head"
        if string_frame == result.executed_path.header.frame_id:
            continue
        for p in result.executed_path.poses:
            ps = tf.transform(p, string_frame).pose.position
            if (distance:=np.sqrt(ps.y**2 + ps.z**2)) < minimum_distance:
                minimum_distance = distance
                closest_neighbor = string

    row = utils.row_from_result(result)
    row['min_distance'] = minimum_distance
    score = o2p.score_row(row)

    out = "Result:\n" + "\n".join([f"{o.note}: {o.loudness:.2F}dBA" for o in result.onsets]) + f"\nSafety Score: {score:.3F}\nMinimum Distance: {minimum_distance:.4F}m (to {closest_neighbor})"
    rospy.loginfo(out)
    o2p.add_sample(row)


if __name__ == "__main__":
    main()
