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
from ruckig import InputParameter, Ruckig, Trajectory

__all__ = [
    "get_path_yz_offsets_yz_start",
    "get_path_yz_start_y_offset_lift_angle",
    "RuckigPath",
]


class RuckigPath:
    actionspace_type = "ruckig keypoint position/velocity v2"

    def __init__(self):
        self.max_vel = (0.1, 0.1)
        self.max_acc = (1.0, 1.5)
        self.max_jerk = (8.0, 8.0)
        self.pre = (0.0, 0.0)
        self.post = (0.0, 0.0)

        self.keypoint_pos = (0.0, 0.0)
        self.keypoint_vel = (0.0, 0.0)

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

    def params(self):
        return (
            self.max_vel +
            self.max_acc +
            self.max_jerk +
            self.pre +
            self.post +
            self.keypoint_pos +
            self.keypoint_vel +
            (self.string_position,)
        )

    def to_action_parameters(self):
        return ActionParameters(
            actionspace_type= self.actionspace_type,
            action_parameters= list(self)
        )

    @classmethod
    def from_action_parameters(cls, msg):
        if msg.actionspace_type != RuckigPath.actionspace_type:
            raise ValueError("Invalid actionspace type")

        m=re.match(r"guzheng/(\w+)/head", msg.header.frame_id)
        if not m:
            raise ValueError("Invalid frame_id")

        params = cls()
        params.note = m.group(1)
        params.max_vel = msg.action_parameters[0:2]
        params.max_acc = msg.action_parameters[2:4]
        params.max_jerk = msg.action_parameters[4:6]
        params.pre = msg.action_parameters[6:8]
        params.post = msg.action_parameters[8:10]
        params.keypoint_pos = msg.action_parameters[10:12]
        params.keypoint_vel = msg.action_parameters[12:14]
        params.string_position = msg.action_parameters[14]
        return params

    @classmethod
    def random(cls, *, note, direction= None, string_position= None, tf= None):
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
            try:
                p= PointStamped()
                p.header.frame_id = f"guzheng/{note}/bridge"
                length = tf.transform(p, f"guzheng/{note}/head").point.x
            except tf2_ros.TransformException as e:
                raise Exception(f"No string position defined and could not find length of target string for note {note}: {str(e)}")
        p.string_position = string_position

        if direction is None:
            direction = random.choice((-1.0, 1.0))

        p.pre = (direction*(-0.015), 0.015)
        p.post = (direction*0.01, 0.02)
        p.keypoint_pos = (random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.001))
        p.keypoint_vel = (direction*random.uniform(0.005, 0.06), random.uniform(0.005, 0.03))

        return p

    @property
    def frame_id(self) -> str:
        return f"guzheng/{self.note}/head"

    def build(self):
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
            ruckig_generator.calculate(inp, t)
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
    
    def dataframe(self):
        return pandas.DataFrame([(p.header.stamp.to_sec(), p.pose.position.y, p.pose.position.z) for p in self.build().poses], columns= ["time", "y", "z"])

    def keypoint_marker(self):
        pk_pos = self.keypoint_pos
        pk_vel = self.keypoint_vel
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
                Point(self.string_position, pk_pos[0], pk_pos[1]),
                Point(self.string_position, pk_pos[0]+pk_vel[0]*pk_vel_scale, pk_pos[1]+pk_vel[1]*pk_vel_scale)
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