#!/usr/bin/env python

import librosa
import matplotlib.pyplot as plt; plt.switch_backend('agg')
import numpy as np
import os.path
import pandas as pd
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
from std_msgs.msg import Bool as BoolMsg

def main():
    rospy.init_node('one_pluck_by_loudness')

    # read parameters
    string = rospy.get_param("~string") # string to explore, might be multiple separated by spaces or "all" for all fitted ones
    strings = string.split(" ")
    finger = rospy.get_param("~finger", "ff") # finger to use to pluck, one of "ff", "mf", "rf", "th"
    target_loudness = rospy.get_param("~loudness") # target loudness

    direction = rospy.get_param("~direction") # direction to pluck in (>0 towards the robot, <0 away from the robot)
    storage = rospy.get_param("~storage", "plucks.json")
    storage_path = rospkg.RosPack().get_path("tams_pr2_guzheng") + f"/data/{storage}"

    # validate parameters
    valid_fingers = ("ff", "mf", "rf", "th")
    if finger not in valid_fingers:
        rospy.logfatal(f"invalid finger '{finger}', use one of {valid_fingers}")
        return

    # only continue if mannequin mode is inactive (state is published on topic)
    if rospy.wait_for_message("mannequin_mode_active", BoolMsg).data:
        rospy.logfatal("mannequin mode is active, aborting")
        return

    o2p= OnsetToPath(storage= storage_path)

    run_episode = SimpleActionClient("run_episode", RunEpisodeAction)
    run_episode.wait_for_server()

    while True:
        target_string = random.choice(strings)
        target_direction = random.choice((-1, 1)) if direction == 0.0 else direction

        rospy.loginfo(f"targeting string {target_string} in direction {target_direction} for loudness {target_loudness:.2F}dBA")
        p = RuckigPath.prototype(string= target_string, direction= target_direction)
        p, finger, _ = o2p.get_path(
            note= utils.string_to_note(target_string),
            finger= finger,
            direction= target_direction,
            loudness= target_loudness,
            )
        run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= p.action_parameters, string= p.string, finger= finger)))
        run_episode.wait_for_result()

        r = utils.row_from_result(run_episode.get_result())
        if r["loudness"] is None:
            r["loudness"] = -1.0
        rospy.loginfo(f'      yielded {r["loudness"]:.2F}dBA')

if __name__ == "__main__":
    main()
