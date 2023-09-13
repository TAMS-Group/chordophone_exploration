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
    rospy.init_node('performance_curve')

    # read parameters
    string = rospy.get_param("~string") # string to explore, might be multiple separated by spaces or "all" for all fitted ones
    finger = rospy.get_param("~finger", "ff") # finger to use to pluck, one of "ff", "mf", "rf", "th"
    direction = rospy.get_param("~direction") # direction to pluck in (>0 towards the robot, <0 away from the robot), 0.0 for random
    string_position = rospy.get_param("~string_position", -1.0) # position on the string to pluck, <0 for random
    if string_position < 0:
        string_position = None
    steps = rospy.get_param("~steps", 20) # number of steps between min and max in storage
    repetitions_per_step = rospy.get_param("~repetitions", 3) # number of trials per step
    storage = rospy.get_param("~storage", "plucks_explore.json")
    storage_path = rospkg.RosPack().get_path("tams_pr2_guzheng") + f"/data/{storage}"
    shuffle = rospy.get_param("~shuffle", True)

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

    min, max = o2p.get_note_min_max(utils.string_to_note(string))

    # load json if it exists, else create new df
    results_path = storage_path.replace(".json", f"_performance_results.json")
    if os.path.exists(results_path):
        results = pd.read_json(results_path).to_dict('records')
        rospy.loginfo(f'loaded existing results with {len(results)} trials')
    else:
        rospy.loginfo('no existing results found, creating new df')
        results = []

    trials_list = np.repeat(np.linspace(min, max, steps, endpoint= True), repetitions_per_step)
    if shuffle:
        np.random.default_rng(37).shuffle(trials_list)

    for step, target_loudness in enumerate(trials_list):
        rospy.loginfo(f"{step}/{len(trials_list)}: targeting loudness {target_loudness:.2F}dBA")
        p = RuckigPath.prototype(string= string, direction= direction)
        p, finger, _ = o2p.get_path(
            note= utils.string_to_note(string),
            finger= finger,
            direction= direction,
            loudness= target_loudness,
            string_position= string_position,
            )
        run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= p.action_parameters, string= p.string, finger= finger)))
        run_episode.wait_for_result()

        r = utils.row_from_result(run_episode.get_result())
        del r['onsets'] # this field is a list of onsets, but pandas just ignores the whole dict if it sees it
        df = pd.DataFrame(r.copy(), columns= r.keys(), index= [0])
        if r['loudness'] is None:
            r['loudness'] = -1.0
        r['safety_score'] = utils.score_safety(df)[0]
        r['target_loudness'] = target_loudness
        r['string_position'] = string_position
        r['shuffle'] = shuffle
        results.append(r)
        rospy.loginfo(f'      yielded {"safe" if r["safety_score"] > 0.0 else "unsafe"} / {r["loudness"]:.2F}dBA')

    # save results
    rospy.loginfo(f"writing results to {results_path}")
    pd.DataFrame(results).to_json(results_path)

if __name__ == "__main__":
    main()
