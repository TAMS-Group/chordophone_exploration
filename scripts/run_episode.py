#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from actionlib import SimpleActionClient

from tams_pr2_guzheng.msg import (
    ExecutePathAction,
    EpisodeState,
    ActionParameters,
    ExecutePathGoal)

import random
import copy

import numpy as np
from math import tau

class RunEpisode():
    def __init__(self, just_play=False):
        self.goto_start_client = SimpleActionClient(
            'pluck/goto_start',
            ExecutePathAction)
        self.goto_start_client.wait_for_server()

        self.execute_path_client = SimpleActionClient(
            'pluck/execute_path',
            ExecutePathAction)
        self.execute_path_client.wait_for_server()

        self.just_play = just_play

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

        # leave time for clients to connect
        rospy.sleep(rospy.Duration(1.0))

        self.episode_id = 0
        self.episode_cnt = 0

    def new_episode(self):
        self.episode_id = random.randint(0, 1 << 30)
        self.episode_cnt+= 1
        rospy.loginfo(f'run episode number {self.episode_cnt}')

    def publishState(self, state, now=None):
        es = EpisodeState()
        es.header.stamp = rospy.Time.now() if now is None else now
        es.state = state
        es.episode = self.episode_id
        self.state_pub.publish(es)

    def sleep(self, t):
        if not self.just_play:
            rospy.sleep(rospy.Duration(t))

    @staticmethod
    def makeParameters(parameter_type, parameters, now=None):
        ap = ActionParameters()
        ap.header.stamp = rospy.Time.now() if now is None else now
        ap.actionspace_type = parameter_type
        ap.action_parameters = parameters
        return ap

    @staticmethod
    def get_path_yz_offsets_yz_start(note):
        y_start = random.uniform(-0.010, 0.000)
        z_start = random.uniform(0.0, 0.010)
#        y_start = -0.015
#        z_start = 0.0
        y_rand = random.uniform(-.010, 0.000)
        z_rand = random.uniform(.0, 0.015)

        # waypoints relative to sampled start
        waypoints = [
            [.05,  0.000 + 0.000,          0.01 + 0.015],
            [.05, -0.006 + 0.000,          0.00 + 0.015],
            [.05, -0.006 + 0.000 + y_rand, 0.00 + 0.015],
            [.05, -0.020 + 0.000 + y_rand, 0.01 + 0.015+z_rand],
            # back to start
            # [.05, 0.00 +0.000,        0.01+0.015]
            ]

        for w in waypoints:
            w[1] += y_start
            w[2] += z_start

        path = Path()
        path.header.frame_id = 'guzheng/{}/head'.format(note)

        for x, y, z in waypoints:
            p = PoseStamped()
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = z
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        return path, RunEpisode.makeParameters(
            "y z waypoint offsets / yz start",
            [y_rand, z_rand, y_start, z_start])

    @staticmethod
    def get_path_yz_start_y_offset_lift_angle(note):
        y_start = random.uniform(-0.010, 0.005)
        z_start = random.uniform(-0.000, 0.005)
#        y_start = -0.015
#        z_start = 0.0
        y_rand = random.uniform(-.010, 0.000)

        lift_rand = random.uniform(tau/10, tau/4)

        lift_dist = 0.02
        lift_wp_y = y_rand - lift_dist * np.cos(lift_rand)
        lift_wp_z = lift_dist * np.sin(lift_rand)

        # waypoints relative to sampled start
        waypoints = [
            [.05,  0.000 + 0.000,          0.01 + 0.015],
            [.05, -0.006 + 0.000,          0.00 + 0.015],
            [.05, -0.006 + 0.000 + y_rand, 0.00 + 0.015],
            [.05, -0.006 + 0.000 + lift_wp_y, 0.00 + 0.015 + lift_wp_z],
            # back to start
            # [.05, 0.00 +0.000,        0.01+0.015]
            ]

        for w in waypoints:
            w[1] += y_start
            w[2] += z_start

        path = Path()
        path.header.frame_id = 'guzheng/{}/head'.format(note)

        for x, y, z in waypoints:
            p = PoseStamped()
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = z
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        return path, RunEpisode.makeParameters(
            "yz start / y offset / lift angle",
            [y_start, z_start, y_rand, lift_rand])

    def run_episode(self, note, repeat=1):
        # path, params = RunEpisode.get_path_yz_offsets_yz_start(note)
        path, params = RunEpisode.get_path_yz_start_y_offset_lift_angle(note)

        finger = 'ff'

        for i in range(repeat):
            self.new_episode()

            approach_path = copy.deepcopy(path)
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
            self.execute_path_client.send_goal(ExecutePathGoal(
                path=path,
                finger=finger
                ))
            params.header.stamp = now
            self.parameter_pub.publish(params)
            self.execute_path_client.wait_for_result()
            # wait to collect data
            self.sleep(2.0)
            self.publishState("end")
            self.sleep(1.0)


def main():
    rospy.init_node('run_episode')

    just_play = rospy.get_param("~just_play", False)
    re = RunEpisode(just_play=just_play)

    note = rospy.get_param("~note", "d6")

    continuous = rospy.get_param("~continuous", False)
    runs = rospy.get_param("~runs", 0)
    repeat = rospy.get_param("~repeat", 1)

    if just_play:
        notes = ["d6", "b5", "a5", "fis5", "e5", "d5", "b4", "a4", "fis4"]
        for n in notes:
            re.run_episode(note=n, repeat=1)
        for n in reversed(notes):
            re.run_episode(note=n, repeat=1)
    elif runs > 0:
        for i in range(runs):
            re.run_episode(note=note, repeat=repeat)
            rospy.sleep(rospy.Duration(1.0))
    elif not continuous:
        re.run_episode(note=note, repeat=repeat)
        rospy.sleep(rospy.Duration(1.0))
    else:
        while not rospy.is_shutdown():
            re.run_episode(note=note, repeat=repeat)


if __name__ == "__main__":
    main()
