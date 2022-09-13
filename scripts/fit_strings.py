#!/usr/bin/env python

import rospy

from tf import transformations
import tf2_ros
from tf2_msgs.msg import TFMessage

from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray

import numpy as np
# import numpy.linalg as linalg

from skimage.measure import LineModelND, ransac

from threading import Lock


class StringFitter:
    def __init__(self):
        self.lock = Lock()
        self.onsets = {}

        self.tf_broadcast = tf2_ros.StaticTransformBroadcaster()

    def start(self):
        self.sub_notes = rospy.Subscriber(
            'guzheng/onsets_projected',
            MarkerArray,
            self.onsets_cb,
            tcp_nodelay=True)

    def onsets_cb(self, msg):
        with self.lock:
            self.onsets = {}
            for m in msg.markers:
                if m.ns != "unknown" and m.ns != "":
                    self.onsets[m.ns] = \
                        self.onsets.get(m.ns, []) + [m.pose.position]

            self.fit()

    def fit(self):
        tf_msg = TFMessage()
        for k in self.onsets.keys():
            if len(self.onsets[k]) >= 5:
                rospy.loginfo(f'fit {k}')
                pts = np.array(
                    [(p.x, p.y, p.z) for p in self.onsets[k]],
                    dtype=float)
                model, inliers = ransac(
                    pts,
                    LineModelND,
                    min_samples=2,
                    residual_threshold=0.01,
                    max_trials=500
                    )
                if np.sum(inliers) < 5:
                    continue
                inlier_pts = pts[inliers]
                tf = TransformStamped()
                tf.header.frame_id = 'base_footprint'
                tf.child_frame_id = k
                tf.transform.translation.x = model.params[0][0]
                tf.transform.translation.y = model.params[0][1]
                tf.transform.translation.z = model.params[0][2]

                direction = model.params[1]
                if direction[1] < 0:
                    direction = -direction

                min_pt = np.min((inlier_pts - model.params[0]) @ model.params[1])
                bridge_pt = model.params[0] + min_pt * direction
                tf.transform.translation.x = bridge_pt[0]
                tf.transform.translation.y = bridge_pt[1]
                tf.transform.translation.z = bridge_pt[2]

                rot = np.zeros((4, 4), float)
                rot[0:3, 0:3] = np.diag((1, 1, 1))
                rot[3, 3] = 1.0

                rot[0:3, 0] = direction
                rot[0:3, 2] = [0.0, 0.0, 1.0]
                rot[0:3, 1] = np.cross(rot[0:3, 0], rot[0:3, 2])
                rot[0:3, 1] /= np.sqrt(np.sum(rot[0:3, 1]**2))
                if rot[0, 1] > 0.0:
                    rot[0:3, 1] = -rot[0:3, 1]
                rot[0:3, 2] = np.cross(rot[0:3, 0], rot[0:3, 1])
                rot_q = transformations.quaternion_from_matrix(rot)
                rot_q = rot_q / np.sqrt(np.sum(rot_q**2))
                tf.transform.rotation.x = rot_q[0]
                tf.transform.rotation.y = rot_q[1]
                tf.transform.rotation.z = rot_q[2]
                tf.transform.rotation.w = rot_q[3]
                rospy.loginfo(model.params)
                tf_msg.transforms.append(tf)
        # crude hack. WTF
        # TODO: implement sendTransform*s* in StaticBroadcaster
        self.tf_broadcast.pub_tf.publish(tf_msg)

        # rospy.signal_shutdown('')


def main():
    rospy.init_node('string_fitter')
    sf = StringFitter()
    sf.start()
    rospy.spin()


if __name__ == '__main__':
    main()
