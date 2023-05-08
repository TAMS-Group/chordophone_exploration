#!/usr/bin/env python

import random
from math import tau

import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from actionlib import SimpleActionClient
import scipy.stats as stats

import tams_pr2_guzheng.paths as paths
from tams_pr2_guzheng.msg import (
    RunEpisodeAction,
    RunEpisodeGoal,
    RunEpisodeRequest)
from tams_pr2_guzheng.utils import string_length


def main():
    rospy.init_node('explore')

    tf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf)

    from std_msgs.msg import String as StringMsg
    say_pub = rospy.Publisher("/say", StringMsg, queue_size= 1, tcp_nodelay= True)
    def say(txt):
        say_pub.publish(txt)

    note = rospy.get_param("~note", "d6")
    finger = rospy.get_param("~finger", "ff")
    direction = rospy.get_param("~direction", None)
    if direction == 0.0:
        direction = None
    string_position = rospy.get_param("~string_position", None)
    if string_position is not None and string_position < 0.0:
        string_position = None
    runs = rospy.get_param("~runs", 1)

    run_episode = SimpleActionClient("run_episode", RunEpisodeAction)
    run_episode.wait_for_server()

    # strings to explore
    #strings= [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
    strings= note.split(" ")

    jump_size= 3 # max size of the jump between two consecutively explored strings
    attempts_for_good_pluck = 4 # max number of attempts to pluck string with one onset

    # uniform sampling of targeted string position
    string_position_sampler = stats.qmc.Halton(d= 1, seed= 37)

    strings_idx= np.arange(len(strings))
    # keep histogram of onsets per string
    onset_hist= np.ones(len(strings))

    # currently explored string
    i= random.randint(0, len(strings)-1)

    while not rospy.is_shutdown():
        rospy.loginfo(f"attempting to pluck string {strings[i]}")
        # "runs" in explore mode is the number of times we try to pluck the string before switching the target string
        for _ in range(runs):
            if string_position is None:
                trial_string_position = stats.qmc.scale(string_position_sampler.random(), 0.0, string_length(strings[i], tf))
            else:
                trial_string_position = string_position
            path = paths.RuckigPath.random(
                note = strings[i],
                direction= direction,
                string_position= trial_string_position,
                #tf = tf
                )
            onsets = []
            for _ in range(attempts_for_good_pluck):
                run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= path.action_parameters, string= path.note, finger= finger)))
                run_episode.wait_for_result()
                if rospy.is_shutdown():
                    return
                onsets = run_episode.get_result().onsets

                if len(onsets) > 0:
                    onset_hist[i]+= 1

                if len(onsets) == 1:
                    break

                if len(onsets) == 0:
                    rospy.logwarn("no onset detected, retry with adapted parameters")
                    # lower and further in the pluck direction
                    path.keypoint_pos[0] += 0.003 * path.direction
                    path.keypoint_pos[1] -= 0.003
                else: # len(onsets) > 1
                    rospy.logwarn(f"multiple onsets detected, but one expected (got {len(onsets)}), retry with adapted parameters")
                    # higher
                    path.keypoint_pos[1] += 0.005
                    # move velocity vector (12/13) up by a bit and clip to avoid changing direction
                    theta = tau/4/2 * path.direction
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    vec = np.array(path.keypoint_vel)
                    vec_rotated = np.dot(rot, vec)
                    if vec_rotated[0] * vec[0] <= 0.0:
                        vec_rotated[0] = 0.0
                    path.keypoint_vel= vec_rotated.tolist()
        # shaped string sampling
        # stay close
        p= stats.norm.pdf(strings_idx, loc= i, scale= 1.0)
        # similar chances for all other strings
        p+= stats.uniform.pdf(strings_idx, loc= 0, scale= 20)
        # avoid explored strings
        p/= onset_hist
        # TODO: could normalize by string length as well, but how important is full geometrical coverage?
        p/= p.sum()
        i = np.random.choice(strings_idx, 1, p= p)[0]

if __name__ == "__main__":
    main()
