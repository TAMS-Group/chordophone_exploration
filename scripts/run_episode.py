#!/usr/bin/env python

import rospy

import tf2_ros
import tf2_geometry_msgs

from actionlib import SimpleActionClient

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Header

from tams_pr2_guzheng.msg import (
    ExecutePathAction,
    EpisodeState,
    ActionParameters,
    ExecutePathGoal,
    RunEpisodeRequest,
    NoteOnset
    )

import random
import copy

import numpy as np
from math import tau

from ruckig import InputParameter, Ruckig, Trajectory

class RunEpisode():
    def __init__(self, explore= False, nosleep= False):
        self.tf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf)

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
        self.nosleep = explore or nosleep

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

        # leave time for clients to connect
        rospy.sleep(rospy.Duration(1.0))

        self.episode_id = 0
        self.episode_cnt = 0

        self.biases = []
        self.systematic_bias = {}
        self.episode_onsets = [[]]
        self.reset()

        rospy.loginfo("startup complete")

    def reset(self):
        self.biases = []
        self.next_systematic_bias()
        self.episode_onsets = [[]]

    def new_episode(self):
        self.episode_id = random.randint(0, 1 << 30)
        self.episode_cnt+= 1
        if self.explore:
            self.episode_onsets.append([])
        rospy.loginfo(f'run episode number {self.episode_cnt}')

    def onset_cb(self, onset):
        if self.explore:
            self.episode_onsets[-1].append(onset)

    def publishState(self, state, now=None):
        es = EpisodeState()
        es.header.stamp = rospy.Time.now() if now is None else now
        es.state = state
        es.episode = self.episode_id
        self.state_pub.publish(es)

    def sleep(self, t):
        if not self.nosleep:
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
    def get_path_yz_start_y_offset_lift_angle(note, params= None):
        if params is None:
            y_start = random.uniform(-0.010, 0.005)
            z_start = random.uniform(-0.000, 0.005)
            # y_start = -0.015
            # z_start = 0.0
            y_rand = random.uniform(-.010, 0.000)

            lift_rand = random.uniform(tau/10, tau/4)
        else:
            y_start, z_start, y_rand, lift_rand = params.action_parameters

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

    def get_path_ruckig(self, note, params= None, direction= 0.0, string_position= -1):
        if params and params.actionspace_type != "ruckig keypoint position/velocity":
            rospy.logfatal(f"found unexpected actionspace type '{params.actionspace_type}'")

        # Create instances: the Ruckig OTG as well as input and output parameters
        cycle_time = 0.04
        ruckig_generator = Ruckig(2, cycle_time)  # DoFs, control cycle
        inp = InputParameter(2)
        def traj_from_input(inp):
            ruckig_generator.reset()
            t= Trajectory(2)
            ruckig_generator.calculate(inp, t)
            return t

        # build randomized parameters if not provided
        if params is None:
            try:
                p= PointStamped()
                p.header.frame_id = f"guzheng/{note}/bridge"
                length = self.tf.transform(p, f"guzheng/{note}/head").point.x
            except tf2_ros.TransformException as e:
                length = 0.1
                rospy.logwarn_throttle_identical(60*30, f"could not find length of target string for note {note}: {str(e)}. Defaulting to {length}m")

            # forward(direction = 1.0) or backward(direction = -1.0) pluck
            if direction == 0.0:
                direction = random.choice((-1.0, 1.0))

            pre = (direction*0.015, 0.015)
            post = (direction*(-0.01), 0.02)

            if string_position < 0.0:
                string_position = random.uniform(0.0, length)

            params = RunEpisode.makeParameters(
                        "ruckig keypoint position/velocity",
                        # TODO: tune defaults
                        [
                        # max velocities
                        0.1, 0.3,
                        # max accelerations
                        1.0, 1.0,
                        # max jerk
                        8.0, 8.0,
                        # pre position
                        self.systematic_bias['y']+pre[0], self.systematic_bias['z']+pre[1],
                        # post position
                        self.systematic_bias['y']+post[0], self.systematic_bias['z']+post[1],
                        # keypoint position
                        self.systematic_bias['y'] + random.uniform(-0.005, 0.005), self.systematic_bias['z'] + random.uniform(-0.005, 0.001),
                        # keypoint velocity
                        direction*random.uniform(-0.06,-0.005), random.uniform(0.005, 0.03),
                        # string position
                        string_position
                        ])

        # parse parameters from input message
        _params = params.action_parameters[:]
        def next_parameter(n):
            nonlocal _params
            p = _params[:n]
            _params = _params[n:]
            return p
        inp.max_velocity = next_parameter(2)
        inp.max_acceleration = next_parameter(2)
        inp.max_jerk = next_parameter(2)
        pre = next_parameter(2)
        post = next_parameter(2)
        keypoint_position = next_parameter(2)
        keypoint_velocity = next_parameter(2)
        head_offset = next_parameter(1)[0]

        # build pluck trajectories
        inp.current_position = pre
        inp.current_velocity = [0.0, 0.0]
        inp.current_acceleration = [0.0, 0.0]
        inp.target_position = keypoint_position
        inp.target_velocity = keypoint_velocity
        inp.target_acceleration = [0.0,0.0]
        t= traj_from_input(inp)
        kp= np.array(t.at_time(t.duration)).T
        inp.current_position = kp[:,0]
        inp.current_velocity = kp[:,1]
        inp.current_acceleration = kp[:,2]
        inp.target_position = post
        inp.target_velocity = [0.0,0.0]
        inp.target_acceleration = [0.0,0.0]
        t2= traj_from_input(inp)
        # sample & combine trajectories
        n_wp= int((t.duration+t2.duration)/cycle_time)
        kp_idx= int((t.duration)/cycle_time)
        T= np.linspace(0.0, t.duration+t2.duration, n_wp)
        Y= np.zeros((n_wp, 2, 3))
        for i in range(0, kp_idx):
            Y[i,:,:] = np.array(t.at_time(T[i])).T
        for i in range(kp_idx, n_wp):
            Y[i,:,:] = np.array(t2.at_time(T[i]-t.duration)).T

        path = Path()
        path.header.frame_id = 'guzheng/{}/head'.format(note)
        for i in range(Y.shape[0]):
            p = PoseStamped()
            p.header.stamp = rospy.Time(T[i])
            p.pose.position.x = head_offset
            p.pose.position.y = float(Y[i, 0, 0])
            p.pose.position.z = float(Y[i, 1, 0])
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        return path, params

    def next_systematic_bias(self):
        if len(self.biases) == 0:
            yr = np.append([0.0], np.random.uniform(-0.01, 0.005, 3))
            self.biases = [{"y": y, "z": 0.0} for y in yr]
        self.systematic_bias = self.biases.pop(0)

    def run_episode(self, note= 'd6', finger= 'ff', repeat=1, params= None, direction= 0.0, string_position= -1):
        # path, params = RunEpisode.get_path_yz_offsets_yz_start(note)
        # path, params = RunEpisode.get_path_yz_start_y_offset_lift_angle(note, params=params)
        path, params = self.get_path_ruckig(note, params=params, direction=direction, string_position=string_position)

        self.finger_pub.publish(finger)

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
            self.pluck_client.send_goal(ExecutePathGoal(
                path=path,
                finger=finger
                ))
            params.header.stamp = now
            self.parameter_pub.publish(params)
            self.pluck_client.wait_for_result()
            # wait to collect data
            self.sleep(2.0)
            self.publishState("end")
            self.sleep(1.0)

            # adapt systematic_bias if required
            if len(self.episode_onsets) >= 4:
                note_notation = note.upper().replace("IS", "♯")
                eos = [[o.note for o in e] for e in self.episode_onsets]
                # did not pluck any strings - go lower
                if len([e for e in self.episode_onsets if len(e) > 0]) < 2 and self.systematic_bias['z'] > -0.01:
                    self.systematic_bias['z']= max((-0.01, self.systematic_bias['z']-0.002))
                    rospy.loginfo(f'did not record enough onsets in {eos}. adapting systematic z bias to {self.systematic_bias["z"]}')
                # did not pluck correct string - try sampled vicinity
                elif len([e for e in self.episode_onsets if len([o for o in e if o.note == note_notation]) > 0]) < 1:
                    self.next_systematic_bias()
                    detected_notes = set()
                    for e in self.episode_onsets:
                        detected_notes.update([o.note for o in e])
                    rospy.loginfo(f'did not recognize target note {note_notation} often enough in {eos}. adapt systematic bias to {self.systematic_bias} to try plucking neighboring strings')
                # if all plucks hit, revert z to 0.0 assuming the ransac model stablized
                elif len([e for e in self.episode_onsets if len(e) > 0]) == 4:
                    self.systematic_bias['z'] *= 0.3
                self.episode_onsets.clear()


def main():
    rospy.init_node('run_episode')

    listen = rospy.get_param("~listen", False)
    explore = rospy.get_param("~explore", False)
    re = RunEpisode(explore= explore, nosleep= listen)

    note = rospy.get_param("~note", "d6")
    finger = rospy.get_param("~finger", "ff")
    direction = rospy.get_param("~direction", 0.0)
    string_position = rospy.get_param("~string_position", -1.0)

    continuous = rospy.get_param("~continuous", False)
    runs = rospy.get_param("~runs", 1)
    repeat = rospy.get_param("~repeat", 1)


    if listen:
        rospy.loginfo("subscribing to topic to wait for action parameter requests")

        def param_cb(msg):
            rospy.loginfo(f"received request for {msg.finger} / {msg.string} / parameters {msg.parameters}")
            re.run_episode(finger= msg.finger, note= msg.string, params= msg.parameters)
        rospy.Subscriber("~", RunEpisodeRequest, param_cb, queue_size= 1)
        rospy.spin()
    elif explore:
        rospy.loginfo("exploring expected strings")

        jump_size= 3
        #strings= [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
        strings= ["a3", "a4"]

        i= random.randint(0, len(strings)-1)
        while not rospy.is_shutdown():
            rospy.loginfo(f"attempting to pluck string {strings[i]}")
            for _ in range(runs):
                re.run_episode(finger= finger, note=strings[i], repeat=repeat)
            re.reset()
            i= max(0, min(len(strings)-1, i+random.randint(-jump_size,jump_size)))
    elif continuous or runs > 0:
        if continuous:
            rospy.loginfo("running continuously")
        else:
            rospy.loginfo(f"running for {runs} episode(s) with {repeat} repetitions each")
        i = 0
        while continuous or i < runs:
            if rospy.is_shutdown():
                break
            re.run_episode(finger= finger, note= note, repeat= repeat, direction= direction, string_position= string_position)
            i+=1
            #rospy.sleep(rospy.Duration(1.0))
    else:
        rospy.logerr("found invalid configuration. Can't go on.")

if __name__ == "__main__":
    main()
