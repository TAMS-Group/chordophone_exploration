#!/usr/bin/env python

import rospy

import tf2_ros
import tf2_geometry_msgs

from actionlib import SimpleActionClient, SimpleActionServer
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker

import tams_pr2_guzheng.paths as paths

from tams_pr2_guzheng.msg import (
    ExecutePathAction,
    EpisodeState,
    ActionParameters,
    ExecutePathGoal,
    RunEpisodeAction,
    RunEpisodeGoal,
    RunEpisodeResult,
)

from music_perception.msg import NoteOnset

import random
import copy

import numpy as np
from scipy.stats import qmc
from math import tau

class RunEpisode():
    def __init__(self, *, tf, nosleep= False):
        self.episode_id = 0
        self.episode_cnt = 0

        self.episode_onsets = []
        self.executed_path = None

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

        self.keypoint_pub = rospy.Publisher(
            'pluck/keypoint',
            Marker,
            queue_size=1,
            tcp_nodelay=True,
            latch=True)

        self.tf = tf

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.onset_sub = rospy.Subscriber(
            'guzheng/onsets_haptically_validated',
            NoteOnset,
            self.onset_cb,
            queue_size=500,
            tcp_nodelay=True
            )

        # leave time for clients to connect
        rospy.sleep(rospy.Duration(1.0))

        rospy.loginfo("startup complete")

    def new_episode(self):
        self.episode_id = int(rospy.Time.now().to_sec())
        self.episode_cnt+= 1

        rospy.loginfo(f'run episode number {self.episode_cnt}')

    def reset_buffers(self):
        self.episode_onsets = []
        self.executed_path = None

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

    def publish_support_msgs(self, path, finger):
        self.keypoint_pub.publish(path.keypoint_marker)

        target_pluck_string = TransformStamped()
        target_pluck_string.header.stamp = rospy.Time.now()
        target_pluck_string.header.frame_id = f"guzheng/{path.string}/head"
        target_pluck_string.child_frame_id = "target_pluck_string"
        target_pluck_string.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(target_pluck_string)
        target_pluck_string.header.stamp += rospy.Duration(6.0)
        self.tf_broadcaster.sendTransform(target_pluck_string)

    def run_episode(self, path, finger= 'ff'):
        self.publish_support_msgs(path, finger)

        self.new_episode()

        p = path()

        # build two-waypoint path to approach start position with some clearance from above
        # TODO: for *long* approach paths, go up first, then towards a higher approach point, then down
        approach_path = copy.deepcopy(p)
        approach_path.poses = approach_path.poses[0:1]
        approach_pose = copy.deepcopy(approach_path.poses[0])
        approach_pose.pose.position.z += 0.025
        approach_path.poses.insert(0, approach_pose)
        self.goto_start_client.send_goal(ExecutePathGoal(
            path=approach_path,
            finger=finger
            ))
        self.goto_start_client.wait_for_result()

        if rospy.is_shutdown():
            return

        self.reset_buffers()

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

        self.finger = finger
        self.executed_path = self.pluck_client.get_result().executed_path

        # wait to collect data
        self.sleep(0.8)
        self.publishState("end")
        # some more buffer to associate messages later if needed
        #self.sleep(1.0)

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
    re = RunEpisode(nosleep= nosleep, tf = tf)

    string = rospy.get_param("~string", "d6")
    finger = rospy.get_param("~finger", "ff")
    direction = rospy.get_param("~direction", 0.0)
    string_position = rospy.get_param("~string_position", -1.0)

    continuous = rospy.get_param("~continuous", False)
    runs = rospy.get_param("~runs", 1)
    repeat = rospy.get_param("~repeat", 1)

    if listen:
        rospy.loginfo("set up action server")

        action_server = None
        def goal_cb(goal : RunEpisodeGoal):
            rospy.loginfo(f"received request for {goal.req.finger} / {goal.req.string} / parameters {goal.req.parameters}")
            path = None
            if goal.req.parameters.actionspace_type != '':
                try:
                    path = paths.RuckigPath.from_action_parameters(goal.req.parameters)
                except Exception as e:
                    rospy.logerr(f"failed to create path from parameters: {e}")
                    action_server.set_aborted()
                    return True
            else:
                path = paths.RuckigPath.random(goal.req.string)
            # TODO: abort if execution fails / frame is not known
            re.run_episode(path, finger= goal.req.finger)
            action_server.set_succeeded(result=
                                        RunEpisodeResult(
                                            onsets= re.episode_onsets,
                                            executed_path= re.executed_path,
                                            parameters= path.action_parameters,
                                            finger= goal.req.finger,
                                        )
            )
            return True

        action_server = SimpleActionServer("run_episode", RunEpisodeAction, execute_cb= goal_cb, auto_start= False)
        action_server.start()
        rospy.loginfo("action server ready")
        rospy.spin()
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
                string = string,
                direction= direction,
                string_position= string_position,
                tf = re.tf
            )

            for i in range(repeat):
                if rospy.is_shutdown():
                    break
                re.run_episode(finger= finger, path= path)
            i+=1
            say("next")
    else:
        rospy.logfatal("found invalid configuration. Can't go on.")

if __name__ == "__main__":
    main()
