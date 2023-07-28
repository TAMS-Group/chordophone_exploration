#!/usr/bin/env python

import librosa
import matplotlib.pyplot as plt; plt.switch_backend('agg')
import numpy as np
import random
import rospkg
import rospy
import scipy.stats as stats
import tams_pr2_guzheng.paths as paths
import tams_pr2_guzheng.utils as utils
import tf2_geometry_msgs
import tf2_ros

from actionlib import SimpleActionClient
from math import tau
from tams_pr2_guzheng.onset_to_path import OnsetToPath
from tams_pr2_guzheng.utils import string_length, publish_figure
from tams_pr2_guzheng.msg import (
    RunEpisodeAction,
    RunEpisodeGoal,
    RunEpisodeRequest)
from visualization_msgs.msg import MarkerArray

def plot_p(strings, p):
    fig = plt.figure(dpi= 150)
    fig.gca().set_title("explore distribution across target strings")
    fig.gca().bar(np.arange(len(strings)), p, tick_label= strings)
    publish_figure("p", fig)

def main():
    rospy.init_node('explore')

    # read parameters

    string = rospy.get_param("~string", "d6") # string to explore, might be multiple separated by spaces or "all" for all fitted ones
    finger = rospy.get_param("~finger", "ff") # finger to use to pluck, one of "ff", "mf", "rf", "th"
    direction = rospy.get_param("~direction", 0.0) # direction to pluck in (>0 towards the robot, <0 away from the robot), 0.0 for random
    string_position = rospy.get_param("~string_position", -1.0) # position on the string to pluck, <0 for random
    runs = rospy.get_param("~runs", 0) # number of total calls to run_episode before terminating, 0 for infinite
    attempts_per_string = rospy.get_param("~attempts_per_string", 1) # number of attempts to pluck a target string before switching to another
    # path strategy to use for exploration. one of
    # "random" - randomly sample all open parameters
    # "reduce_variance" - sample string position and keypoint_pos_y from a distribution that minimizes GP variance
    # "geometry" - retry variations if sample comes out without detected onset
    strategy = rospy.get_param("~strategy", "random")
    # strategy to sample string position from. one of
    # "uniform" - uniform sampling
    # "halton" - halton sequence sampling
    position_strategy = rospy.get_param("~position_strategy", "halton")
    # max number of attempts to pluck string with one onset with strategy=="geometry"
    attempts_for_good_pluck = rospy.get_param("~attempts_for_good_pluck", 4)
    # storage path for explored plucks
    storage_path = rospy.get_param("~storage", rospkg.RosPack().get_path("tams_pr2_guzheng") + "/data/plucks_explore.json")

    # validate parameters
    if attempts_per_string < 1:
        rospy.logfatal("attempts_per_string must be >= 1")
        return

    valid_fingers = ("ff", "mf", "rf", "th")
    if finger not in valid_fingers:
        rospy.logfatal(f"invalid finger '{finger}', use one of {valid_fingers}")
        return

    valid_position_strategies = ("halton", "uniform")
    if position_strategy not in valid_position_strategies:
        rospy.logfatal(f"invalid position_strategy '{position_strategy}', use one of {valid_position_strategies}")
        return

    valid_strategies = ("random", "reduce_variance", "geometry")
    if strategy not in valid_strategies:
        rospy.logfatal(f"invalid strategy '{strategy}', use one of {valid_strategies}")
        return

    # prepare exploration
    if strategy != "geometry":
        o2p= OnsetToPath(storage= storage_path) # throws on invalid path

    from std_msgs.msg import String as StringMsg
    say_pub = rospy.Publisher("/say", StringMsg, queue_size= 1, tcp_nodelay= True)
    def say(txt):
        say_pub.publish(txt)

    tf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf)

    run_episode = SimpleActionClient("run_episode", RunEpisodeAction)
    run_episode.wait_for_server()

    # strings to explore
    # guzheng:
    # # strings= [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
    if string == "all":
        strings = rospy.wait_for_message("guzheng/fitted_strings", MarkerArray)
        strings = sorted([m.ns for m in strings.markers if " " not in m.ns and len(m.ns) > 0], key = lambda s: librosa.note_to_midi((utils.string_to_note(s))))
    else:
        strings= string.split(" ")

    # uniform sampling of targeted string position
    if position_strategy == "uniform":
        string_position_sampler = stats.uniform(loc= 0.0, scale= 1.0)
    elif position_strategy == "halton":
        string_position_sampler = stats.qmc.Halton(d= 1, seed= 37)

    strings_idx= np.arange(len(strings))
    # keep histogram of onsets per string
    onset_hist= np.ones(len(strings))

    # target string index (sampled in loop)
    i= -1

    say("starting exploration")

    current_run = 0
    while not rospy.is_shutdown() and (runs == 0 or current_run < runs):
        if current_run % attempts_per_string == 0:
            # shaped string sampling
            p = np.zeros(len(strings))
            if i >= 0:
                # prefer neighborhood of previous string
                p+= stats.norm.pdf(strings_idx, loc= i, scale= 1.0)
            # similar chances for all other strings
            p+= stats.uniform.pdf(strings_idx, loc= 0, scale= 20)
            # penalize previously explored strings
            p/= onset_hist
            # TODO: could normalize by string length as well, but how important is full geometrical coverage?
            p/= p.sum()
            plot_p(strings, p)
            i = np.random.choice(strings_idx, 1, p= p)[0]

        current_run+= 1
        rospy.loginfo(f"run {current_run}{'/'+str(runs) if runs > 0 else ''} targeting string {strings[i]}")

        trial_string_position = string_position
        if trial_string_position < 0.0:
            string_len = 0.0
            try:
                string_len = string_length(strings[i], tf)
            except Exception as e:
                rospy.logwarn(e)
                break
            trial_string_position = stats.qmc.scale(string_position_sampler.random(), 0.0, string_len)

        if strategy == "random":
            path = paths.RuckigPath.random(
                string = strings[i],
                direction= direction,
                string_position= trial_string_position
            )
        else:
            if direction == 0.0:
                trial_direction = random.choice((-1.0, 1.0)) # TODO: consider NBP of either direction

            path = paths.RuckigPath.prototype(
                string = strings[i],
                direction= trial_direction,
                string_position= trial_string_position
            )

        if strategy == "geometry":
            path.keypoint_pos[1] = -0.002 # more cautious initial pluck

        if strategy == "reduce_variance":
            nbp = o2p.infer_next_best_pluck(
                string= strings[i],
                finger= finger,
                direction= path.direction,
                actionspace= OnsetToPath.ActionSpace(
                    string_position= np.array((0.0, string_len)),
                    keypoint_pos_y= np.array((-0.007, 0.008)),
                ),
            )
            if nbp is not None:
                path.string_position = nbp[0]
                path.keypoint_pos[0] = nbp[1]

        # TODO: adjust launch file to new parameter meanings!
        for _ in range(attempts_for_good_pluck):
            run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= path.action_parameters, string= path.string, finger= finger)))
            run_episode.wait_for_result()
            if rospy.is_shutdown():
                break
            result = run_episode.get_result()

            expected_onset = [o for o in result.onsets if o.note == utils.string_to_note(strings[i])]

            # if len(expected_onset) > 1:
            #     # some strings "echo" with a high delay, so we only keep the one with highest loudness to supress artifacts
            #     expected_onset = [max(expected_onset, key= lambda o: o.loudness)]

            unexpected_onsets = [o for o in result.onsets if o.note != utils.string_to_note(strings[i])]

            if strategy != "geometry":
                log = f"add pluck for string {strings[i]} "
                warn = False

                if len(unexpected_onsets) > 0:
                    log+= f"with unexpected onsets {', '.join([o.note for o in unexpected_onsets])} as 0.0dB "
                    warn = True
                    # technically an invalid pluck (or wrong classification...)
                    # treat it as a pluck with no onset to avoid it later on
                    expected_onset = []
                    result.onsets = []
                elif len(expected_onset) > 0:
                    log+= f"with perceived note '{expected_onset[0].note}' ({expected_onset[0].loudness:.2F}dB) "
                else:
                    log+= "without onset "
                log+= "to table"
                if warn:
                    rospy.logwarn(log)
                else:
                    rospy.loginfo(log)
                o2p.add_sample(utils.row_from_result(result))

            if len(result.onsets) > 0:
                onset_hist[i]+= 1

            if strategy != "geometry" or (len(expected_onset) > 0 and len(unexpected_onsets) == 0):
                break

            if len(result.onsets) == 0:
                rospy.logwarn("no onset detected, retry with adapted parameters")
                # lower and further in the pluck direction
                path.keypoint_pos[0] += 0.003 * path.direction
                path.keypoint_pos[1] -= 0.002
            else: # len(onsets) > 1
                rospy.logwarn(f"multiple onsets detected, but one expected (got {len(result.onsets)}), retry with adapted parameters")
                # higher
                path.keypoint_pos[0] *= 0.5
                path.keypoint_pos[1] += 0.005
                # move velocity vector (12/13) up by a bit and clip to avoid changing direction
                theta = tau/4/2 * path.direction
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                vec = np.array(path.keypoint_vel)
                vec_rotated = np.dot(rot, vec)
                if vec_rotated[0] * vec[0] <= 0.0:
                    vec_rotated[0] = 0.0
                path.keypoint_vel= vec_rotated.tolist()
    say("stopped")

if __name__ == "__main__":
    main()
