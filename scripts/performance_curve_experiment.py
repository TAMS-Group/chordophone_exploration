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
    string = rospy.get_param("~string") # string to explore, might be multiple separated by spaces
    finger = rospy.get_param("~finger", "ff") # finger to use to pluck, one of "ff", "mf", "rf", "th"
    direction = float(rospy.get_param("~direction")) # direction to pluck in (>0 towards the robot, <0 away from the robot), 0.0 for random
    string_position = rospy.get_param("~string_position", -1.0) # position on the string to pluck, 0 for both
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

    # load json if it exists, else create new df
    results_path = storage_path.replace(".json", f"_performance_results.json")
    if os.path.exists(results_path):
        results_df = pd.read_json(results_path)
        results = results_df.to_dict('records')
        rospy.loginfo(f'loaded existing results with {len(results)} trials')
    else:
        rospy.loginfo('no existing results found, creating new df')
        results_df = pd.DataFrame()
        results = []

    trials = pd.DataFrame(
        list(zip(*[
            x.ravel() for x in np.meshgrid(
                string.split(" "),
                [direction] if direction != 0.0 else [-1.0, 1.0],
                np.linspace(0.0, 1.0, steps, endpoint=True),
                np.arange(0, repetitions_per_step, 1)
            )
        ])),
        columns= [
            "string",
            "direction",
            "target_loudness_fraction",
            "repetition"
        ]
    )

    if shuffle:
        trials = trials.sample(frac=1, random_state= 37).reset_index(drop=True)

    for i, trial in enumerate(trials.itertuples()):
        loud_min, loud_max = o2p.get_note_min_max(utils.string_to_note(trial.string), trial.direction)
        target_loudness = trial.target_loudness_fraction * (loud_max - loud_min) + loud_min
        rospy.loginfo(f"{i}/{len(trials)}: {trial.string} {trial.direction} {target_loudness:.2F} {trial.repetition}")

        p = RuckigPath.prototype(string= trial.string, direction= trial.direction)
        p, finger, _ = o2p.get_path(
            note= utils.string_to_note(trial.string),
            finger= finger,
            direction= trial.direction,
            loudness= target_loudness,
            string_position= string_position,
            )
        run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= p.action_parameters, string= p.string, finger= finger)))
        run_episode.wait_for_result()
        r = utils.row_from_result(run_episode.get_result())

        del r['onsets'] # this field is a list of onsets, but pandas just ignores the whole dict if it sees it
        df = pd.DataFrame(r.copy(), columns= r.keys(), index= [0])
        r['safety_score'] = utils.score_safety(df)[0]
        if r['loudness'] is None:
            r['loudness'] = -1.0
        r['target_loudness'] = target_loudness
        r['string_position'] = string_position
        r['shuffle'] = shuffle
        r['direction'] = direction
        results.append(r)
        rospy.loginfo(f'      yielded {"safe" if r["safety_score"] > 0.0 else "unsafe"} / {r["loudness"]:.2F}dBA')
        pd.DataFrame(results).to_json(results_path+".tmp")


    # save results
    rospy.loginfo(f"writing results to {results_path}")
    pd.DataFrame(results).to_json(results_path)

if __name__ == "__main__":
    main()
