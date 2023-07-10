#!/usr/bin/env python

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

    tf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf)

    from std_msgs.msg import String as StringMsg
    say_pub = rospy.Publisher("/say", StringMsg, queue_size= 1, tcp_nodelay= True)
    def say(txt):
        say_pub.publish(txt)

    string = rospy.get_param("~string", "d6")
    finger = rospy.get_param("~finger", "ff")
    direction = rospy.get_param("~direction", None)
    if direction == 0.0:
        direction = None
    string_position = rospy.get_param("~string_position", None)
    if string_position is not None and string_position < 0.0:
        string_position = None
    runs = rospy.get_param("~runs", 1)
    attempts = rospy.get_param("~attempts", 0)

    reduce_variance = rospy.get_param("~reduce_variance", False)
    o2p= OnsetToPath(storage= rospy.get_param("~storage", rospkg.RosPack().get_path("tams_pr2_guzheng") + "/data/plucks_explore.json"))

    run_episode = SimpleActionClient("run_episode", RunEpisodeAction)
    run_episode.wait_for_server()

    # strings to explore
    #strings= [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
    if string == "all":
        strings = rospy.wait_for_message("guzheng/fitted_strings", MarkerArray)
        strings = [m.ns for m in strings.markers if " " not in m.ns]
    else:
        strings= string.split(" ")

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
        # rospy.loginfo(f"attempting to pluck string {strings[i]}")
        # "runs" in explore mode is the number of times we try to pluck the string before switching the target string
        for _ in range(runs):
            trial_string_position = string_position
            if trial_string_position is None:
                string_len = 0.0
                try:
                    string_len = string_length(strings[i], tf)
                except Exception as e:
                    rospy.logwarn(e)
                    break
                trial_string_position = stats.qmc.scale(string_position_sampler.random(), 0.0, string_len)

            path = paths.RuckigPath.random(
                string = strings[i],
                direction= direction,
                string_position= trial_string_position
            )

            if reduce_variance:
                nbp = o2p.infer_next_best_pluck(
                    string= strings[i],
                    finger= finger,
                    direction= path.direction,
                    actionspace= OnsetToPath.ActionSpace(
                        string_position= np.array((0.0, string_len)),
                        keypoint_pos_y= np.array((-0.004, 0.008)),
                    ),
                )
                if nbp is not None:
                    path.string_position = nbp[0]
                    path.keypoint_pos[0] = nbp[1]

            for _ in range(attempts_for_good_pluck):
                run_episode.send_goal(RunEpisodeGoal(RunEpisodeRequest(parameters= path.action_parameters, string= path.string, finger= finger)))
                run_episode.wait_for_result()
                if rospy.is_shutdown():
                    return
                result = run_episode.get_result()

                # if more than one onset with expected note exists, only keep the one with highest loudness
                expected_onset = [o for o in result.onsets if o.note == utils.string_to_note(strings[i])]
                if len(expected_onset) > 1:
                    # some strings "echo" with a high delay, so we only keep the one with highest loudness to supress artifacts
                    # TODO: get rid of this through tactile validation
                    expected_onset = [max(expected_onset, key= lambda o: o.loudness)]
                unexpected_onsets = [o for o in result.onsets if o.note != utils.string_to_note(strings[i])]

                log = f"add pluck for string {strings[i]} "
                warn = False

                if len(unexpected_onsets) > 0:
                    log+= f"with unexpected onsets {', '.join([o.note for o in unexpected_onsets])} as 0.0dB "
                    warn = True
                    # technically an invalid pluck (or wrong classification...)
                    # treat it as a pluck with no onset to avoid it later on
                    expected_onset = []
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

                # either we have a good pluck by now or we recorded a bad pluck explicitly in o2p and leave the next parameters to the model
                if reduce_variance or len(result.onsets) == 1:
                    break

                if len(result.onsets) == 0:
                    rospy.logwarn("no onset detected, retry with adapted parameters")
                    # lower and further in the pluck direction
                    path.keypoint_pos[0] += 0.003 * path.direction
                    path.keypoint_pos[1] -= 0.003
                else: # len(onsets) > 1
                    rospy.logwarn(f"multiple onsets detected, but one expected (got {len(result.onsets)}), retry with adapted parameters")
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

        if attempts == 1:
            break
        if attempts > 1:
            attempts-= 1
            rospy.loginfo(f"{attempts} more attempts")

        # shaped string sampling
        # stay close
        p= stats.norm.pdf(strings_idx, loc= i, scale= 1.0)
        # similar chances for all other strings
        p+= stats.uniform.pdf(strings_idx, loc= 0, scale= 20)
        # avoid explored strings
        p/= onset_hist
        # TODO: could normalize by string length as well, but how important is full geometrical coverage?
        p/= p.sum()
        plot_p(strings, p)
        i = np.random.choice(strings_idx, 1, p= p)[0]

if __name__ == "__main__":
    main()
