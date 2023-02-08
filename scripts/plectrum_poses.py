#!/usr/bin/env python

import rospy

from tf import transformations
import tf2_ros
from tf2_msgs.msg import TFMessage

from geometry_msgs.msg import TransformStamped

import numpy as np

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
        self.tf_broadcast = tf2_ros.StaticTransformBroadcaster()

    def run(self):
        # crude hack. WTF
        # TODO: implement sendTransform*s* in StaticBroadcaster
        tf_msg = TFMessage(
            transforms=[tf_for_finger(f) for f in self.fingers])
        self.tf_broadcast.pub_tf.publish(tf_msg)
        rospy.spin()

def main():
    rospy.init_node('plectrum_poses')
    poses = PlectrumPoses()
    poses.run()


if __name__ == '__main__':
    main()
