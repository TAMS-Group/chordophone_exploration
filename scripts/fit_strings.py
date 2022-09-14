#!/usr/bin/env python

import rospy

from tf import transformations
import tf2_ros
from tf2_msgs.msg import TFMessage

from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA

import numpy as np

from skimage.measure import LineModelND, ransac

from threading import Lock


# TODO read pluck events and improve ransac fit with them
class StringFitter:
    def __init__(self):
        self.lock = Lock()
        self.onsets = {}

        self.tf_broadcast = tf2_ros.StaticTransformBroadcaster()
        self.pub_strings = rospy.Publisher(
            'guzheng/fitted_strings',
            MarkerArray,
            queue_size=1,
            tcp_nodelay=True,
            latch=True)

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

    @staticmethod
    def string_to_tf(string):
        tf = TransformStamped()
        tf.header.frame_id = 'base_footprint'
        tf.child_frame_id = string["key"]

        tf.transform.translation.x = string["bridge"][0]
        tf.transform.translation.y = string["bridge"][1]
        tf.transform.translation.z = string["bridge"][2]

        rot = np.zeros((4, 4), float)
        rot[0:3, 0:3] = np.diag((1, 1, 1))
        rot[3, 3] = 1.0

        rot[0:3, 0] = string["direction"]
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

        return tf

    @staticmethod
    def string_to_marker(string):
        m = Marker()
        m.ns = string["key"]
        m.id = 0
        m.action = Marker.ADD
        m.type = Marker.CYLINDER
        m.header.frame_id = 'base_footprint'
        m.scale.x = m.scale.y = 0.003
        m.scale.z = np.sqrt(np.sum((string["end"]-string["bridge"])**2))
        m.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)

        center = (string["end"]-string["bridge"])/2 + string["bridge"]
        m.pose.position.x = center[0]
        m.pose.position.y = center[1]
        m.pose.position.z = center[2]

        rot = np.zeros((4, 4), float)
        rot[0:3, 0:3] = np.diag((1, 1, 1))
        rot[3, 3] = 1.0

        rot[0:3, 2] = string["direction"]
        rot[0:3, 1] = [0.0, 0.0, 1.0]
        rot[0:3, 0] = np.cross(rot[0:3, 0], rot[0:3, 2])
        rot[0:3, 0] /= np.sqrt(np.sum(rot[0:3, 0]**2))
        if rot[0, 0] > 0.0:
            rot[0:3, 0] = -rot[0:3, 0]
        rot[0:3, 1] = np.cross(rot[0:3, 2], rot[0:3, 0])
        rot_q = transformations.quaternion_from_matrix(rot)
        rot_q = rot_q / np.sqrt(np.sum(rot_q**2))
        m.pose.orientation.x = rot_q[0]
        m.pose.orientation.y = rot_q[1]
        m.pose.orientation.z = rot_q[2]
        m.pose.orientation.w = rot_q[3]

        return m

    @staticmethod
    def align_bridges(strings):
        origins = np.array([s["bridge"] for s in strings], dtype=float)

        # TODO: do not assume projection plane is with normal (0,0,1)
        # this is only true for guzheng and other lying chordophones
        def project(o):
            return o[:, 0:2]

        origins2d = project(origins)
        model, inliers = ransac(
            origins2d,
            LineModelND,
            min_samples=2,
            residual_threshold=0.01,
            max_trials=1000
            )
        rospy.loginfo(f'{np.sum(inliers)} inliers for bridge fitting')

        model_origin = model.params[0]
        model_direction = model.params[1]

        for s in strings:
            # solve line intersection between ransaced bridge and string
            # in 2d projection space
            _, t = np.linalg.solve(
                np.array([
                    s["direction"][0:2],
                    -model_direction], dtype=float).T,
                model_origin - s["bridge"][0:2])
            intersect = model_origin + t * model_direction

            # adapt bridge along string in 3d
            s["bridge"] += s["direction"] * (
                (intersect - s["bridge"][0:2]) @ s["direction"][0:2]
                )

        return strings

    def fit(self):
        strings = []
        for k in sorted(self.onsets.keys()):
            if len(self.onsets[k]) >= 5:
                pts = np.array(
                    [(p.x, p.y, p.z) for p in self.onsets[k]],
                    dtype=float)
                model, inliers = ransac(
                    pts,
                    LineModelND,
                    min_samples=2,
                    residual_threshold=0.01,
                    max_trials=1000
                    )
                rospy.loginfo(
                    f'fit {k} with {len(pts)} points '
                    f'({np.sum(inliers)} inliers)')
                if np.sum(inliers) < 5:
                    rospy.loginfo('skipped because of few inliers')
                    continue

                origin = model.params[0]
                direction = model.params[1]

                inlier_pts = pts[inliers]

                if direction[1] < 0:
                    direction = -direction

                min_pos_on_string = \
                    np.min((inlier_pts - model.params[0]) @ direction)
                bridge_pt = origin + min_pos_on_string * direction

                max_pos_on_string = \
                    np.max((inlier_pts - model.params[0]) @ direction)
                end_pt = model.params[0] + max_pos_on_string * direction

                strings.append({
                    "key": k.replace("♯", "is"),
                    "bridge": bridge_pt,
                    "direction": direction,
                    "end": end_pt
                    })

        StringFitter.align_bridges(strings)

        # crude hack. WTF
        # TODO: implement sendTransform*s* in StaticBroadcaster
        tf_msg = TFMessage(
            transforms=[StringFitter.string_to_tf(s) for s in strings])
        self.tf_broadcast.pub_tf.publish(tf_msg)

        markers = MarkerArray(
            markers=[StringFitter.string_to_marker(s) for s in strings])
        self.pub_strings.publish(markers)


def main():
    rospy.init_node('string_fitter')
    sf = StringFitter()
    sf.start()
    rospy.spin()


if __name__ == '__main__':
    main()
