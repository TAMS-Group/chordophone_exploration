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
    RunEpisodeRequest,
    ChordophoneEstimation,
    )
from std_msgs.msg import Bool as BoolMsg

def plot_p(strings, p, chosen= None):
    fig = plt.figure(dpi= 150)
    fig.gca().set_title("explore distribution across target strings")
    fig.gca().bar(np.arange(len(strings)), p, tick_label= strings)
    # color chosen string in fitting complementary color
    if chosen is not None:
        fig.gca().get_children()[chosen].set_color("darkred")
    publish_figure("p", fig)

def main():
    rospy.init_node('explore')

    # read parameters

    string = rospy.get_param("~string", "d6") # string to explore, might be multiple separated by spaces or "all" for all fitted ones
    initial_string = rospy.get_param("~initial_string", "") # string to start with when multiple are specified
    finger = rospy.get_param("~finger", "ff") # finger to use to pluck, one of "ff", "mf", "rf", "th"
    direction = rospy.get_param("~direction", 0.0) # direction to pluck in (>0 towards the robot, <0 away from the robot), 0.0 for random
    string_position = rospy.get_param("~string_position", -1.0) # position on the string to pluck, <0 for random
    runs = rospy.get_param("~runs", 0) # number of total calls to run_episode before terminating, 0 for infinite
    attempts_per_string = rospy.get_param("~attempts_per_string", 1) # number of attempts to pluck a target string before switching to another
    # path strategy to use for exploration. one of
    # "random" - randomly sample all open parameters
    # "avpe" - Active Valid Pluck Exploration. sample string position and keypoint_pos_y from a distribution that minimizes GP variance
    # "geometry" - retry variations if sample comes out without detected onset
    strategy = rospy.get_param("~strategy", "random")
    # strategy to sample string position from. one of
    # "uniform" - uniform sampling
    # "halton" - halton sequence sampling
    position_strategy = rospy.get_param("~position_strategy", "halton")
    # max number of attempts to pluck string with one onset with strategy=="geometry"
    attempts_for_good_pluck = rospy.get_param("~attempts_for_good_pluck", 4)
    # storage path for explored plucks
    storage = rospy.get_param("~storage", "")
    if storage != "":
        storage = "_"+str(storage)
    storage_path = rospkg.RosPack().get_path("tams_pr2_guzheng") + f"/data/plucks_explore{storage}.json"

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

    valid_strategies = ("random", "avpe", "geometry")
    if strategy not in valid_strategies:
        rospy.logfatal(f"invalid strategy '{strategy}', use one of {valid_strategies}")
        return

    # only continue if mannequin mode is inactive (state is published on topic)
    try:
        if rospy.wait_for_message("mannequin_mode_active", BoolMsg, timeout= 10.0).data:
            rospy.logfatal("mannequin mode is active, aborting")
            return
    except rospy.ROSException:
        rospy.loginfo("could not check for mannequin mode, continuing")
        pass

    # prepare exploration

    if strategy != "geometry":
        o2p= OnsetToPath(storage= storage_path) # throws on invalid path

    tf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf)

    run_episode = SimpleActionClient("run_episode", RunEpisodeAction)
    run_episode.wait_for_server()

    # strings to explore
    # guzheng:
    # # strings= [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
    fitted_strings = rospy.wait_for_message("guzheng/estimate", ChordophoneEstimation)
    known_strings = sorted([s.key for s in fitted_strings.strings], key = lambda s: librosa.note_to_midi(utils.string_to_note(s)))
    if string == "all":
        strings = known_strings
    else:
        strings= string.split(" ")

    # uniform sampling of targeted string position
    if position_strategy == "uniform":
        string_sampler = lambda i, d=stats.uniform(loc= 0.0, scale= 1.0): d.rvs()
    elif position_strategy == "halton":
        # separate sampler for each string is essential
        halton_sequences = [stats.qmc.Halton(d= len(strings), seed= 37) for _ in range(len(strings))]
        string_sampler = lambda i, sequences = halton_sequences: sequences[i].random()[:, i:i+1]  # retain 2d shape (for scale) with dimensions 1x1

    uniform_sampler = lambda d=stats.uniform(loc= 0.0, scale= 1.0): np.array([[d.rvs()]])

    strings_idx= np.arange(len(strings))
    # keep histogram of onsets per string
    onset_hist= np.ones(len(strings))

    string_len = 0.0
    actionspace = RuckigPath.ActionSpace(
        string_position= np.array((0.0, string_len)),
        keypoint_pos_y= np.array((-0.01, 0.018)),
        keypoint_pos_z= np.array((-0.004,)),
        keypoint_vel_y= np.array((0.015,)),
        keypoint_vel_z= np.array((0.015,)),
    )

    # target string index (sampled in loop)
    if initial_string in strings:
        i = strings.index(initial_string)
    else:
        i = -1

    say("starting exploration")

    trial_direction = 1.0
    current_run = 0
    while not rospy.is_shutdown() and (runs == 0 or current_run < runs):
        # sample new target string
        if current_run % attempts_per_string == 0:
            # shaped string sampling
            p = np.zeros(len(strings))
            if i >= 0:
                # prefer neighborhood of previous string
                p+= 3*stats.norm.pdf(strings_idx, loc= i, scale= 2.0)
            # similar chances for all other strings
            p+= 0.03*stats.uniform.pdf(strings_idx, loc= 0, scale= 20)
            # penalize previously explored strings
            p/= 10*onset_hist
            # we could normalize by string length as well, but geometric coverage turns out to be less important
            p/= p.sum()

            i = np.random.choice(strings_idx, 1, p= p)[0]

            plot_p(strings, p, chosen= i)

            # update actionspace.string_position
            try:
                string_len = string_length(strings[i], tf)
            except Exception as e:
                rospy.logwarn(e)
                break
            if string_position < 0.0:
                actionspace = actionspace._replace(string_position = np.array((0.0, string_len)))
            else:
                if string_position > string_len:
                    rospy.logwarn(f"string_position {string_position} is larger than string length {string_len}, clipping")
                    new_string_position = string_len
                else:
                    new_string_position = string_position
                actionspace = actionspace._replace(string_position = np.array((new_string_position,)))

        # TODO: consider NBP of either direction in infer_next_best_pluck
        if strategy == "geometry" or current_run % attempts_per_string == 0:
            trial_direction = random.choice((-1.0, 1.0)) if direction == 0.0 else direction


        current_run+= 1
        rospy.loginfo(f"run {current_run}{'/'+str(runs) if runs > 0 else ''} targeting string {strings[i]}")

        # prepare path depending on strategy


        path = RuckigPath.prototype(
            string = strings[i],
            direction= trial_direction,
        )

        if strategy == "geometry":
            # start with a slightly higher pluck in geometry exploration
            # if the string is missed, the follow-up attempts will be lower
            path.keypoint_pos[1] += 0.002
            path.string_position = stats.qmc.scale(string_sampler(i), *actionspace.string_position)
        elif strategy == "random":
            path.sample(actionspace, uniform_sampler)
        elif strategy == "avpe":
            path = o2p.infer_next_best_pluck(
                string= strings[i],
                finger= finger,
                direction= path.direction,
                actionspace= actionspace.with_direction(path.direction),
            )
        else:
            rospy.logfatal(f"invalid strategy '{strategy}'")
            return

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
                # compute minimum distance to other strings during execution (used in validity score)
                minimum_distance = np.inf
                closest_neighbor = "none"
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

                # compact log output and eventually add result to o2p
                log = f"add pluck for string {strings[i]} "
                logger = {'info' : rospy.loginfo, 'warn' : rospy.logwarn, 'error' : rospy.logerr}
                level = "info"
                if len(unexpected_onsets) > 0:
                    log+= f"with unexpected onsets {', '.join([o.note for o in unexpected_onsets])} "
                elif len(expected_onset) > 0:
                    log+= f"with perceived note '{expected_onset[0].note}' ({expected_onset[0].loudness:.2F}dB) "
                else:
                    log+= "without onset "
                log+= f"and score {score:.3F} (dist {minimum_distance:.4F}m to {closest_neighbor}) "
                log+= f"and neighborhood_context {row['neighborhood_context']:.3F} "
                log+= "to table"

                if score < 0.0:
                    level= "warn"
                logger[level](log)

                o2p.add_sample(row)

            if len(result.onsets) > 0:
                onset_hist[i]+= 1

            if strategy != "geometry" or (len(expected_onset) > 0 and len(unexpected_onsets) == 0):
                break
            if len(unexpected_onsets) > 0 and len(expected_onset) == 0:
                rospy.logwarn(f"unexpected onsets detected: {', '.join([o.note for o in unexpected_onsets])}. Skipping retry")
                break
            if len(result.onsets) == 0:
                rospy.logwarn("no onset detected, retry with adapted parameters")
                # lower and further in the pluck direction
                path.keypoint_pos[0] += 0.003 * path.direction
                path.keypoint_pos[1] -= 0.001
            else: # len(onsets) > 1
                rospy.logwarn(f"multiple onsets detected, but one expected (got {len(result.onsets)}), retry with adapted parameters")
                # higher
                path.keypoint_pos[0] -= 0.002 * path.direction
                path.keypoint_pos[1] += 0.001
                # move velocity vector (12/13) up by a bit and clip to avoid changing direction
                theta = tau/4 / 2 * path.direction
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                vec = np.array(path.keypoint_vel)
                vec_rotated = np.dot(rot, vec)
                if vec_rotated[0] * vec[0] <= 0.0:
                    vec_rotated[0] = 0.0
                path.keypoint_vel= vec_rotated.tolist()
    say("stopped")

if __name__ == "__main__":
    main()
