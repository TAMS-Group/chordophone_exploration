import rospy

import tf2_ros
import tf2_geometry_msgs

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Vector3, Pose, Quaternion
from std_msgs.msg import String, Header, ColorRGBA
from visualization_msgs.msg import Marker

from tams_pr2_guzheng.msg import (
    ActionParameters
)

import random
import re

import numpy as np
from math import tau

import pandas

import ruckig
from ruckig import InputParameter, Ruckig, Trajectory

from .utils import string_length

__all__ = [
    "get_path_yz_offsets_yz_start",
    "get_path_yz_start_y_offset_lift_angle",
    "RuckigPath",
]


class RuckigPath:
    actionspace_type = "ruckig keypoint position/velocity v2"

    def __init__(self):
        self.max_vel = [0.1, 0.1]
        self.max_acc = [1.0, 1.5]
        self.max_jerk = [8.0, 8.0]
        self.pre = [0.0, 0.0]
        self.post = [0.0, 0.0]

        self.keypoint_pos = [0.0, 0.0]
        self.keypoint_vel = [0.0, 0.0]

        self.note = "None"
        self.string_position = 0.0

    def __str__(self) -> str:
        return (
            f"RuckigPath(\n"
            f"    note= {self.note}\n"
            f"    string_position= {self.string_position:.3f}\n"
            f"    keypoint_pos= {self.keypoint_pos[0]:.3f} {self.keypoint_pos[1]:.3f}\n"
            f"    keypoint_vel= {self.keypoint_vel[1]:.4f} {self.keypoint_vel[1]:.4f}\n"
            f")"
        )

    @property
    def params(self):
        p = []
        p += self.max_vel
        p += self.max_acc
        p += self.max_jerk
        p += self.pre
        p += self.post
        p += self.keypoint_pos
        p += self.keypoint_vel
        p += [self.string_position]
        return p

    @property
    def params_map(self):
        return {
            "note": self.note,
            "max_vel_y": self.max_vel[0],
            "max_vel_z": self.max_vel[1],
            "max_acc_y": self.max_acc[0],
            "max_acc_z": self.max_acc[1],
            "max_jerk_y": self.max_jerk[0],
            "max_jerk_z": self.max_jerk[1],
            "pre_y": self.pre[0],
            "pre_z": self.pre[1],
            "post_y": self.post[0],
            "post_z": self.post[1],
            "keypoint_pos_y": self.keypoint_pos[0],
            "keypoint_pos_z": self.keypoint_pos[1],
            "keypoint_vel_y": self.keypoint_vel[0],
            "keypoint_vel_z": self.keypoint_vel[1],
            "string_position": self.string_position,
            }

    @property
    def action_parameters(self):
        return ActionParameters(
            header= Header(frame_id=self.frame_id),
            actionspace_type= self.actionspace_type,
            action_parameters= self.params,
        )

    @classmethod
    def from_action_parameters(cls, msg):
        if msg.actionspace_type != RuckigPath.actionspace_type:
            raise ValueError("Invalid actionspace type")

        m=re.match(r"guzheng/(\w+)/head", msg.header.frame_id)
        if not m:
            raise ValueError(f"Invalid frame_id '{msg.header.frame_id}'")

        params = cls()
        params.note = m.group(1)

        msg_parameters = list(msg.action_parameters)
        params.max_vel = msg_parameters[0:2]
        params.max_acc = msg_parameters[2:4]
        params.max_jerk = msg_parameters[4:6]
        params.pre = msg_parameters[6:8]
        params.post = msg_parameters[8:10]
        params.keypoint_pos = msg_parameters[10:12]
        params.keypoint_vel = msg_parameters[12:14]
        params.string_position = msg_parameters[14]
        return params

    @classmethod
    def random(cls, *, note : str, direction : float = None, string_position : float = None, tf = None):
        '''
        @param note: note to play
        @param direction: -1.0 (away from robot) or 1.0 (towards robot). Random if None.
        @param string_position: pluck position on string in meters. Random on string if None and tf are given
        '''

        if string_position is None and tf is None:
            raise ValueError("Must provide either string_position or tf")

        p = __class__()

        p.note = note
        if string_position is None:
            string_position = random.uniform(0.0, string_length(tf, note))
        p.string_position = string_position

        if direction is None:
            direction = random.choice((-1.0, 1.0))

        p.pre = [direction*(-0.015), 0.015]
        p.post = [direction*0.01, 0.02]
        p.keypoint_pos = [random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.001)]
        p.keypoint_vel = [direction*random.uniform(0.005, 0.06), random.uniform(0.005, 0.03)]

        return p

    @property
    def feasible(self):
        df= self.dataframe
        if not np.all(np.abs(df['y'].to_numpy()) < 0.016):
            rospy.logwarn_throttle(1.0, f"Trajectory is not feasible: might hit neighboring string")
            return False
        if not np.all(df['z'].to_numpy() > -0.015):
            rospy.logwarn_throttle(1.0, f"Trajectory is not feasible: finger would touch string")
            return False
        return True

    def __call__(self):
        '''
        @return: path message
        '''
        # Create instances: the Ruckig OTG as well as input and output parameters
        cycle_time = 0.04
        ruckig_generator = Ruckig(2, cycle_time)  # DoFs, control cycle
        inp = InputParameter(2)
        inp.max_velocity = self.max_vel
        inp.max_acceleration = self.max_acc
        inp.max_jerk = self.max_jerk

        def traj_from_input(inp):
            ruckig_generator.reset()
            t= Trajectory(2)
            result = ruckig_generator.calculate(inp, t)
            if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
                raise Exception('Failed to generate trajectory. Invalid input')
            return t

        # build pluck trajectories
        inp.current_position = self.pre
        inp.current_velocity = (0.0, 0.0)
        inp.current_acceleration = (0.0, 0.0)
        inp.target_position = self.keypoint_pos
        inp.target_velocity = self.keypoint_vel
        inp.target_acceleration = (0.0,0.0)
        t= traj_from_input(inp)
        
        kp= np.array(t.at_time(t.duration)).T
        inp.current_position = kp[:,0]
        inp.current_velocity = kp[:,1]
        inp.current_acceleration = kp[:,2]
        inp.target_position = self.post
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
        path.header.frame_id = self.frame_id
        for i in range(Y.shape[0]):
            p = PoseStamped()
            p.header.stamp = rospy.Time(T[i])
            p.pose.position.x = self.string_position
            p.pose.position.y = float(Y[i, 0, 0])
            p.pose.position.z = float(Y[i, 1, 0])
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        return path

    @property
    def frame_id(self) -> str:
        return f"guzheng/{self.note}/head"

    @property
    def direction(self) -> float:
        return self.post[0]/abs(self.post[0])

    @property
    def dataframe(self):
        return pandas.DataFrame([(p.header.stamp.to_sec(), p.pose.position.y, p.pose.position.z) for p in self().poses], columns= ["time", "y", "z"])

    @property
    def keypoint_marker(self):
        pk_pos = np.array(self.keypoint_pos)
        pk_vel = np.array(self.keypoint_vel)
        pk_vel_scale = 0.25
        # publish arrow marker for pk_*
        return Marker(
            header=Header(
                frame_id=self.frame_id,
                stamp=rospy.Time.now()
            ),
            pose=Pose(position=Point(), orientation=Quaternion(0,0,0,1)),
            type=Marker.ARROW,
            action=Marker.ADD,
            scale=Vector3(0.001, 0.003, 0.0),
            color=ColorRGBA(1.0, 0.0, 0.0, 1.0),
            points=[
                Point(self.string_position, *pk_pos),
                Point(self.string_position, *(pk_pos + pk_vel_scale*pk_vel))
            ]
        )


def _make_parameters(parameter_type, parameters, now=None):
    ap = ActionParameters()
    ap.header.stamp = rospy.Time.now() if now is None else now
    ap.actionspace_type = parameter_type
    ap.action_parameters = parameters
    return ap

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

    return path, _make_parameters(
        "y z waypoint offsets / yz start",
        [y_rand, z_rand, y_start, z_start])

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

    return path, _make_parameters(
        "yz start / y offset / lift angle",
        [y_start, z_start, y_rand, lift_rand])