#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import actionlib
from tams_pr2_guzheng.msg import ExecutePathAction, ExecutePathGoal, ExecutePathResult

import random

def send_goal(client):
    y_rand = random.uniform(-.01, 0.005)
    z_rand = random.uniform(.0, 0.01)
    waypoints = [
        [.05, 0.00 +0.000,        0.01+0.015],
        [.05,-0.015+0.000,        0.00+0.015],
        [.05,-0.020+0.000+y_rand, 0.00+0.015],
        [.05,-0.025+0.000+y_rand, 0.02+0.015+z_rand],
        [.05, 0.00 +0.000,        0.01+0.015]
        ]

    path = Path()
    path.header.frame_id = 'guzheng/d6/head'

    for x, y, z in waypoints:
        p = PoseStamped()
        p.pose.position.x = x  # 0 - 0.15
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation.w = 1.0
        path.poses.append(p)
    client.send_goal(ExecutePathGoal(path= path))

def main():
    rospy.init_node('publish_path')
    client = actionlib.SimpleActionClient('pluck/execute_path', ExecutePathAction)
    client.wait_for_server()

    if not rospy.get_param("~continuous", False):
        send_goal(client)
        client.wait_for_result()
    else:
        while not rospy.is_shutdown():
            send_goal(client)
            client.wait_for_result()


if __name__ == "__main__":
    main()
