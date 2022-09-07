#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import actionlib
from tams_pr2_guzheng.msg import ExecutePathAction, ExecutePathGoal, ExecutePathResult, EpisodeState, ActionParameters

import random
import copy

class RunEpisode():
    def __init__(self):
        self.goto_start_client = actionlib.SimpleActionClient('pluck/goto_start', ExecutePathAction)
        self.goto_start_client.wait_for_server()

        self.execute_path_client = actionlib.SimpleActionClient('pluck/execute_path', ExecutePathAction)
        self.execute_path_client.wait_for_server()

        self.state_pub = rospy.Publisher('episode/state', EpisodeState, queue_size=10, tcp_nodelay= True)
        self.parameter_pub = rospy.Publisher('episode/action_parameters', ActionParameters , queue_size=10, tcp_nodelay= True)

        # leave time for clients to connect
        rospy.sleep(rospy.Duration(1.0))

        self.episode_id= 0

    def new_episode(self):
        self.episode_id= random.randint(0, 1<<30)

    def publishState(self, state, now= None):
        es= EpisodeState()
        es.header.stamp = rospy.Time.now() if now is None else now
        es.state = state
        es.episode = self.episode_id
        self.state_pub.publish(es)

    @staticmethod
    def makeParameters(parameter_type, parameters, now= None):
        ap= ActionParameters()
        ap.header.stamp = rospy.Time.now() if now is None else now
        ap.actionspace_type= parameter_type
        ap.action_parameters= parameters
        return ap

    def get_path(self, note):
        y_rand = random.uniform(-.01, 0.005)
        z_rand = random.uniform(.0, 0.01)

        waypoints = [
            [.05, 0.00 +0.000,        0.01+0.015],
            [.05,-0.015+0.000,        0.00+0.015],
            [.05,-0.020+0.000+y_rand, 0.00+0.015],
            [.05,-0.025+0.000+y_rand, 0.02+0.015+z_rand],
#            [.05, 0.00 +0.000,        0.01+0.015]
            ]

        path = Path()
        path.header.frame_id = 'guzheng/{}/head'.format(note)

        for x, y, z in waypoints:
            p = PoseStamped()
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = z
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        return path, RunEpisode.makeParameters("y z waypoint offsets", [y_rand, z_rand])

    def run_episode(self, note, repeat= 1):
        path, params = RunEpisode.get_path(self, note)

        finger= 'ff'

        for i in range(repeat):
            self.new_episode()

            approach_path= copy.deepcopy(path)
            approach_path.poses= approach_path.poses[0:1]
            approach_pose = copy.deepcopy(approach_path.poses[0])
            approach_pose.pose.position.z+= 0.020
            approach_path.poses.insert(0, approach_pose)
            self.goto_start_client.send_goal(ExecutePathGoal(path= approach_path, finger= finger))
            self.goto_start_client.wait_for_result()

            if rospy.is_shutdown():
                return

            now = rospy.Time.now()
            self.publishState("start", now)
            self.execute_path_client.send_goal(ExecutePathGoal(path= path, finger= finger))
            self.parameter_pub.publish(params)
            self.execute_path_client.wait_for_result()
            # wait to collect data
            rospy.sleep(rospy.Duration(2.0))
            self.publishState("end")
            rospy.sleep(rospy.Duration(1.0))

def main():
    rospy.init_node('run_episode')

    re= RunEpisode()

    note= rospy.get_param("~note", "d6")

    repeat= rospy.get_param("~repeat", 1)

    if not rospy.get_param("~continuous", False):
        re.run_episode(note= note, repeat= repeat)
        rospy.sleep(rospy.Duration(1.0))
    else:
        while not rospy.is_shutdown():
            re.run_episode(note= note, repeat= repeat)


if __name__ == "__main__":
    main()
