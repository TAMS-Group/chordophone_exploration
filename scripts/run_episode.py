#!/usr/bin/env python

import rospy

import tf2_ros
import tf2_geometry_msgs

from actionlib import SimpleActionClient

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Vector3, Pose, Quaternion
from std_msgs.msg import String, Header, ColorRGBA
from visualization_msgs.msg import Marker

import tams_pr2_guzheng.paths as paths

from tams_pr2_guzheng.msg import (
    ExecutePathAction,
    EpisodeState,
    ActionParameters,
    ExecutePathGoal,
    RunEpisodeRequest,
)

from music_perception.msg import NoteOnset

import random
import copy

import numpy as np
from math import tau

from ruckig import InputParameter, Ruckig, Trajectory

class RunEpisode():
    def __init__(self, explore= False, nosleep= False):
        rospy.loginfo("connect to execute_path action")
        self.goto_start_client = SimpleActionClient(
            'pluck/execute_path',
            ExecutePathAction)
        self.goto_start_client.wait_for_server()

        rospy.loginfo("connect to pluck action")
        self.pluck_client = SimpleActionClient(
            'pluck/pluck',
            ExecutePathAction)
        self.pluck_client.wait_for_server()

        self.explore = explore
        self.nosleep = nosleep

        self.state_pub = rospy.Publisher(
            'episode/state',
            EpisodeState,
            queue_size=10,
            tcp_nodelay=True)
        self.parameter_pub = rospy.Publisher(
            'episode/action_parameters',
            ActionParameters,
            queue_size=10,
            tcp_nodelay=True)
        self.finger_pub = rospy.Publisher(
            'pluck/active_finger',
            String,
            queue_size=10,
            tcp_nodelay=True)

        self.onset_sub = rospy.Subscriber(
            'guzheng/onsets',
            NoteOnset,
            self.onset_cb,
            queue_size=500,
            tcp_nodelay=True
            )

        self.keypoint_pub = rospy.Publisher(
            'pluck/keypoint',
            Marker,
            queue_size=1,
            tcp_nodelay=True)

        # leave time for clients to connect
        rospy.sleep(rospy.Duration(1.0))

        self.episode_id = 0
        self.episode_cnt = 0

        self.episode_onsets = []

        rospy.loginfo("startup complete")

    def new_episode(self):
        self.episode_id = int(rospy.Time.now().to_sec())
        self.episode_cnt+= 1
        self.episode_onsets = []

        rospy.loginfo(f'run episode number {self.episode_cnt}')

    def onset_cb(self, onset):
        self.episode_onsets.append(onset)

    def publishState(self, state, now=None):
        es = EpisodeState()
        es.header.stamp = rospy.Time.now() if now is None else now
        es.state = state
        es.episode = self.episode_id
        self.state_pub.publish(es)

    def sleep(self, t):
        if not self.nosleep:
            rospy.sleep(rospy.Duration(t))

    def run_episode(self, path, finger= 'ff'):
        self.finger_pub.publish(finger)
        self.keypoint_pub.publish(path.keypoint_marker)

        self.new_episode()

        p = path()

        # build two-waypoint path to approach start position with some clearance from above
        approach_path = copy.deepcopy(p)
        approach_path.poses = approach_path.poses[0:1]
        approach_pose = copy.deepcopy(approach_path.poses[0])
        approach_pose.pose.position.z += 0.020
        approach_path.poses.insert(0, approach_pose)
        self.goto_start_client.send_goal(ExecutePathGoal(
            path=approach_path,
            finger=finger
            ))
        self.goto_start_client.wait_for_result()

        if rospy.is_shutdown():
            return 

        now = rospy.Time.now()
        self.publishState("start", now)
        self.pluck_client.send_goal(ExecutePathGoal(
            path=p,
            finger=finger
            ))
        params = path.action_parameters
        params.header.stamp = now
        self.parameter_pub.publish(params)
        self.pluck_client.wait_for_result()
        # wait to collect data
        self.sleep(2.0)
        self.publishState("end")
        # some more buffer to associate messages later if needed
        if not self.explore:
            self.sleep(1.0)

        return self.episode_onsets

def main():
    rospy.init_node('run_episode')

    tf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf)

    from std_msgs.msg import String as StringMsg
    say_pub = rospy.Publisher("/say", StringMsg, queue_size= 1, tcp_nodelay= True)
    def say(txt):
        say_pub.publish(txt)

    listen = rospy.get_param("~listen", False)
    nosleep = rospy.get_param("~nosleep", False)
    explore = rospy.get_param("~explore", False)
    re = RunEpisode(explore= explore, nosleep= nosleep)

    note = rospy.get_param("~note", "d6")
    finger = rospy.get_param("~finger", "ff")
    direction = rospy.get_param("~direction", None)
    if direction == 0.0:
        direction = None
    string_position = rospy.get_param("~string_position", None)
    if string_position < 0.0:
        string_position = None

    continuous = rospy.get_param("~continuous", False)
    runs = rospy.get_param("~runs", 1)
    repeat = rospy.get_param("~repeat", 1)

    if listen:
        rospy.loginfo("subscribing to topic to wait for action parameter requests")

        def param_cb(msg):
            rospy.loginfo(f"received request for {msg.finger} / {msg.string} / parameters {msg.parameters}")
            path = None
            try:
                path = paths.RuckigPath.from_action_parameters(msg.parameters)
            except Exception as e:
                rospy.logerr(f"failed to create path from parameters: {e}")
                return

            re.run_episode(finger= msg.finger, path= path)

        rospy.Subscriber("~", RunEpisodeRequest, param_cb, queue_size= 1)
        rospy.spin()
    elif explore:
        rospy.loginfo("exploring expected strings")

        # strings to explore
        #strings= [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
        strings= [note]

        jump_size= 3 # max size of the jump between two consecutively explored strings
        attempts_for_good_pluck = 4 # max number of attempts to pluck string with one onset

        i= random.randint(0, len(strings)-1)
        while not rospy.is_shutdown():
            rospy.loginfo(f"attempting to pluck string {strings[i]}")
            # "runs" in explore mode is the number of times we try to pluck the string before switching the target string
            for _ in range(runs):
                path = paths.RuckigPath.random(
                    note = strings[i],
                    direction= direction,
                    string_position= string_position,
                    tf = tf
                    )
                onsets = []
                for _ in range(attempts_for_good_pluck):
                    onsets = re.run_episode(finger= finger, path= path)

                    if rospy.is_shutdown():
                        return

                    if len(onsets) == 1:
                        break

                    rospy.logwarn("retry with adapted parameters")
                    if len(onsets) == 0:
                        # lower and further in the pluck direction
                        path.keypoint_pos[0] += 0.003 * path.direction
                        path.keypoint_pos[1] -= 0.003
                    else: # len(result["onsets"]) > 1
                        # higher
                        path.keypoint_pos[1] += 0.003
                        # move velocity vector (12/13) up by a bit and clip to avoid changing direction
                        theta = tau/4/4 * path.direction
                        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                        vec = np.array(path.keypoint_vel)
                        vec_rotated = np.dot(rot, vec)
                        if vec_rotated[0] * vec[0] < 0.0:
                            vec_rotated[0] = 0.0
                        path.keypoint_vel= vec_rotated.tolist()

            new_i= max(0, min(len(strings)-1, i+random.randint(-jump_size,jump_size)))
            if new_i != i:
                i = new_i

    elif continuous or runs > 0:
        if continuous:
            rospy.loginfo("running continuously")
        else:
            rospy.loginfo(f"running for {runs} episode(s) with {repeat} repetitions each")
        i = 0
        while continuous or i < runs:
            if rospy.is_shutdown():
                break
            path = paths.RuckigPath.random(
                note = strings[i],
                direction= direction,
                string_position= string_position,
                tf = tf
            )

            for i in range(repeat):
                if rospy.is_shutdown():
                    break
                onsets = re.run_episode(finger= finger, path= path)
            i+=1
            say("next")
    else:
        rospy.logfatal("found invalid configuration. Can't go on.")

if __name__ == "__main__":
    main()
