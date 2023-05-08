#!/usr/bin/env python

import copy
import re
from threading import Lock

import numpy as np
import rospkg
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from skimage.measure import LineModelND
from skimage.measure import ransac
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, Empty as EmptySrv
from tf import transformations
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class StringFitter:
    def __init__(self):
        self.tf_broadcast = tf2_ros.StaticTransformBroadcaster()
        self.pub_strings = rospy.Publisher(
            'fitted_strings',
            MarkerArray,
            queue_size=1,
            tcp_nodelay=True,
            latch=True)

        self.strings = []
        self.storage_path = \
            rospy.get_param('~storage_path',
                            rospkg.RosPack().get_path('tams_pr2_guzheng')
                            + '/config/strings.yaml')

        self.active= True
        self.enable_srv = rospy.Service('~set_active', SetBool, self.set_active)
        self.load_from_file_srv = rospy.Service('~load_from_file', EmptySrv, self.load_from_file)
        self.store_to_file_srv = rospy.Service('~store_to_file', EmptySrv, self.store_to_file)

    def start(self):
        self.sub_notes = rospy.Subscriber(
            'onsets_projected',
            MarkerArray,
            self.onsets_cb,
            tcp_nodelay=True,
            queue_size= 1,
            )

    def set_active(self, req):
        self.active = req.data
        if self.active:
            rospy.loginfo("activated fitter")
        else:
            self.publish_strings()
            rospy.loginfo("set fitter inactive after final fit")
        return {'success': True, 'message' : '' }

    def load_from_file(self, req):
        with open(self.storage_path, 'r') as f:
            self.active = False
            plain_strings = yaml.safe_load(f)
            for s in plain_strings:
                s['key'] = str(s['key'])
                s['bridge'] = np.array(s['bridge'])
                s['direction'] = np.array(s['direction'])
                s['end'] = np.array(s['end'])
                s['length'] = float(s['length'])
            self.strings = plain_strings
            rospy.loginfo(f"loaded strings from file {self.storage_path}. Set inactive to avoid overwriting them.")
            self.publish_strings()
        return {}

    def store_to_file(self, req):
        plain_strings = copy.deepcopy(self.strings)
        for s in plain_strings:
            s['bridge'] = [float(x) for x in s['bridge']]
            s['direction'] = [float(x) for x in s['direction']]
            s['end'] = [float(x) for x in s['end']]
            s['length'] = float(s['length'])
        with open(self.storage_path, 'w') as f:
            yaml.dump(plain_strings, f, sort_keys= False)
            rospy.loginfo(f"stored strings to file {self.storage_path}")
        return {}

    def onsets_cb(self, msg):
        if not self.active:
            return

        self.onsets = {}
        for m in msg.markers:
            if m.ns != "unknown" and m.ns != "":
                self.onsets[m.ns] = \
                    self.onsets.get(m.ns, []) + [m.pose.position]
        self.fit()
        self.publish_strings()
        #self.sub_notes.unregister()

    @staticmethod
    def string_to_tfs(string):
        tf = TransformStamped()
        tf.header.frame_id = 'base_footprint'
        tf.child_frame_id = "guzheng/"+string["key"]+"/head"

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

        tf_bridge = TransformStamped()
        tf_bridge.header.frame_id = tf.child_frame_id
        tf_bridge.transform.translation.x = string["length"]
        tf_bridge.transform.rotation.w = 1.0
        tf_bridge.child_frame_id = "guzheng/"+string["key"]+"/bridge"

        return (tf, tf_bridge)

    @staticmethod
    def string_to_marker(string):
        m = Marker()
        m.ns = string["key"]
        m.id = 0
        m.action = Marker.ADD
        m.type = Marker.CYLINDER
        m.header.frame_id = 'base_footprint'
        m.frame_locked = True
        m.scale.x = m.scale.y = 0.003
        m.scale.z = np.sqrt(np.sum((string["end"]-string["bridge"])**2))
        # color a strings green by convention
        if re.match("a[0-9]+", string["key"]):
            m.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
        else:
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

    def project(self, o):
        # TODO: do not assume projection plane is with normal (0,0,1)
        # this is only true for guzheng and other lying chordophones
        # instead project to plane of two principle components of onsets
        # guzheng
        if len(o.shape) > 1:
            return o[:,(0,1)]
        else:
            return np.array(o)[[0,1]]
        # # harp
        # if len(o.shape) > 1:
        #     return o[:,(0,2)]
        # else:
        #     return np.array(o)[[0,2]]

    def split_unexpected_strings(self, strings):
        expected_keys = [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
        expected_strings = [s for s in strings if s["key"] in expected_keys]
        unexpected_strings = [s for s in strings if s["key"] not in expected_keys]
        if len(unexpected_strings) > 0:
            rospy.logwarn(f"drop unexpected strings {[s['key'] for s in unexpected_strings]}")
        return (expected_strings, unexpected_strings)

    def split_unaligned_strings(self, strings):
        s_threshold = 2.0
        # s_threshold = 0.5
        directions = self.project(np.array([s["direction"] for s in strings]))
        angles = np.arctan2(directions[:,0], directions[:,1])
        # Modified Z-score
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        def score(data):
            m= np.median(data)
            d= m-data
            mad= np.median(np.abs(d))
            return 0.6745*d/mad
        s= np.abs(score(angles))
        keys= np.array([s['key'] for s in strings])

        strings_aligned = np.array([s for s in strings], dtype=object)[s < s_threshold].tolist()
        strings_unaligned = np.array([s for s in strings], dtype=object)[s >= s_threshold].tolist()

        if len(strings_unaligned) > 0:
            rospy.loginfo(
                f"found outliers: {', '.join([s['key'] for s in strings_unaligned])}\n"
                f"mod-Z-scores: {', '.join([f'{k}({s:2f})' for (k, s) in sorted(zip(keys, s), key=lambda x: -x[1])])}"
            )

        return (strings_aligned, strings_unaligned)

    def align_bridges(self, strings):
        if len(strings) < 5:
            return []

        aligned_strings = copy.deepcopy(strings)

        origins = np.array([s["bridge"] for s in aligned_strings], dtype=float)

        origins2d = self.project(origins)
        model, inliers = ransac(
            origins2d,
            LineModelND,
            min_samples=2,
            residual_threshold=0.01,
            max_trials=1000
            )
        rospy.loginfo(f'bridge fit with {np.sum(inliers)} inliers from {len(aligned_strings)} strings')

        model_origin = model.params[0]
        model_direction = model.params[1]

        for s in aligned_strings:
            # solve line intersection between ransaced bridge and string
            # in 2d projection space
            _, t = np.linalg.solve(
                np.array([
                    self.project(s["direction"]),
                    -model_direction], dtype=float).T,
                model_origin - self.project(s["bridge"]))
            intersect = model_origin + t * model_direction

            # adapt bridge along string in 3d
            s["bridge"] += s["direction"] * (
                (intersect - self.project(s["bridge"])) @ self.project(s["direction"])
                )

        return aligned_strings

    def fit(self):
        strings = []
        onsets = self.onsets

        log_info = ""
        for k in sorted(onsets.keys()):
            if len(onsets[k]) >= 2:
                pts = np.array(
                    [(p.x, p.y, p.z) for p in onsets[k]],
                    dtype=float)

                inlier_threshold= 0.01

                # special case because the library is broken
                # https://github.com/scikit-image/scikit-image/pull/6755
                if pts.shape[0] == 2:
                    model = LineModelND()
                    model.estimate(pts)
                    inliers = model.residuals(pts) < inlier_threshold
                else:
                    # includes a final model fit on all inliers
                    model, inliers = ransac(
                        pts,
                        LineModelND,
                        min_samples=2,
                        residual_threshold=inlier_threshold,
                        max_trials=1000
                        )
                log_info += (
                    f'fit {k} with {len(pts)} points ({np.sum(inliers)} inliers)\n'
                )

                origin = model.params[0]
                direction = model.params[1]

                inlier_pts = pts[inliers]

                # flip direction to face left for horizontal strings
                if np.abs(direction[1]) > 1.5*np.abs(direction[2]) and direction[1] < 0:
                    direction = -direction

                # flip direction to face upwards for vertical strings
                if np.abs(direction[2]) > 1.5*np.abs(direction[1]) and direction[2] < 0:
                    direction = -direction

                inlier_positions_on_string = (inlier_pts - model.params[0]) @ direction
                min_pos_on_string = np.min(inlier_positions_on_string)
                bridge_pt = origin + min_pos_on_string * direction

                max_pos_on_string = np.max(inlier_positions_on_string)
                end_pt = model.params[0] + max_pos_on_string * direction

                length = max_pos_on_string-min_pos_on_string

                if length < 0.05:
                    log_info += f"skipping very short string for note {k}\n"
                    continue

                strings.append({
                    "key": k.replace("♯", "is").lower(),
                    "bridge": bridge_pt,
                    "direction": direction,
                    "end": end_pt,
                    "length": length
                    })

        rospy.loginfo(log_info)

        self.strings = strings

    def publish_strings(self):
        strings = self.strings
        markers = MarkerArray(markers= [Marker(action = Marker.DELETEALL)])

        strings, unexpected_strings = self.split_unexpected_strings(strings)
        unexpected_strings_markers= [StringFitter.string_to_marker(s) for s in unexpected_strings]
        for m in unexpected_strings_markers:
            m.color = ColorRGBA(0.6,0.6,0.6, 0.3)
            m.ns = "unexpected "+m.ns
        markers.markers.extend(unexpected_strings_markers)

        unaligned_strings= []
        if len(strings) > 5:
            strings, unaligned_strings = self.split_unaligned_strings(strings)
            unaligned_strings_markers = [StringFitter.string_to_marker(s) for s in unaligned_strings]
            for m in unaligned_strings_markers:
                m.color = ColorRGBA(0.8,0.1,0.1, 0.5)
                m.ns = "unaligned "+m.ns
            markers.markers.extend(unaligned_strings_markers)

            bridged_strings = self.align_bridges(strings)
            bridged_strings_markers = [StringFitter.string_to_marker(s) for s in bridged_strings]
            for m in bridged_strings_markers:
                m.scale.x = m.scale.y = m.scale.x/2
                m.ns = "aligned bridge " + m.ns
            markers.markers.extend(bridged_strings_markers)

            # use bridged strings when setting inactive
            if not self.active:
                strings = bridged_strings

        # crude hack. WTF
        # TODO: implement sendTransform*s* in StaticBroadcaster
        tf_msg = TFMessage(
            transforms=[t for s in strings+unaligned_strings for t in StringFitter.string_to_tfs(s)])
        self.tf_broadcast.pub_tf.publish(tf_msg)

        markers.markers.extend([StringFitter.string_to_marker(s) for s in strings])
        self.pub_strings.publish(markers)

def main():
    rospy.init_node('string_fitter')
    sf = StringFitter()
    sf.start()
    rospy.spin()


if __name__ == '__main__':
    main()
