#!/usr/bin/env python

import rospy

from tf import transformations
import tf2_ros

from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from tams_pr2_guzheng.cfg import OffsetsConfig

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

import numpy as np

import copy

def tf_for_finger(finger):
    tf = TransformStamped()
    tf.header.frame_id = f'rh_{finger}_biotac_link'
    tf.child_frame_id = f'rh_{finger}_plectrum'

    tf.transform.translation.x = 0.024
    tf.transform.translation.y = 0.0
    tf.transform.translation.z = -0.01

    rot_q = transformations.quaternion_from_matrix(np.diag(tuple([1.0]*4)))
    tf.transform.rotation.x = rot_q[0]
    tf.transform.rotation.y = rot_q[1]
    tf.transform.rotation.z = rot_q[2]
    tf.transform.rotation.w = rot_q[3]

    return tf


class PlectrumPoses:
    def __init__(self):
        self.fingers = ["th", "ff", "mf", "rf"]
        self.tfs = [tf_for_finger(f) for f in self.fingers]
        self.tf_broadcast = tf2_ros.StaticTransformBroadcaster()
        self.dr_servers= dict()
        for f in self.fingers:
            finger= copy.copy(f)
            cb= lambda finger: lambda c,lvl: self.offset_cb(finger,c,lvl)
            self.dr_servers[f]= DynamicReconfigureServer(OffsetsConfig, cb(f), namespace=f)
        self.publish()

    def offset_cb(self, finger, config, level):
        tf= self.tfs[self.fingers.index(finger)].transform
        tf.translation.x = config.offset_x
        tf.translation.y = config.offset_y
        tf.translation.z = config.offset_z
        self.publish()
        return config

    def publish(self):
        # crude hack. WTF
        # TODO: implement sendTransform*s* in StaticBroadcaster
        self.tf_broadcast.pub_tf.publish(TFMessage(transforms=self.tfs))


def main():
    rospy.init_node('plectrum_poses')
    poses = PlectrumPoses()
    rospy.spin()


if __name__ == '__main__':
    main()
