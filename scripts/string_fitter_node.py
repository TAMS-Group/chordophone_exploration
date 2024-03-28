#!/usr/bin/env python

import copy
import numpy as np
import rospkg
import rospy
import yaml

from geometry_msgs.msg import PoseArray
from skimage.measure import LineModelND
from skimage.measure import ransac
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import SetBool, Empty as EmptySrv
from tams_pr2_guzheng.msg import ChordophoneEstimation
from tams_pr2_guzheng.utils import note_to_string, String
from typing import List
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class StringFitter:
    def __init__(self):
        self.pub_strings = rospy.Publisher(
            'estimate',
            ChordophoneEstimation,
            queue_size=1,
            tcp_nodelay=True,
            latch=True)

        self.pub_string_markers = rospy.Publisher(
            '~estimate_debug',
            MarkerArray,
            queue_size=1,
            tcp_nodelay=True,
            latch=True)

        self.pub_head_poses = rospy.Publisher(
            '~string_head_poses',
            PoseArray,
            queue_size=1,
            tcp_nodelay=True,
            latch=True)

        self.strings = []
        self.storage_path = \
            rospy.get_param('~storage_path',
                            rospkg.RosPack().get_path('tams_pr2_guzheng')
                            + '/data/strings.yaml')

        if rospy.get_param('~load_static_on_startup', False):
            self.load_from_file()
        else:
            self.active= True
        self.enable_srv = rospy.Service('~set_active', SetBool, self.set_active)

        self.exclude_short_strings = rospy.get_param('~exclude_short_strings', True)
        self.exclude_short_strings_srv = rospy.Service('~exclude_short_strings', SetBool, self.set_exclude_short_strings)

        self.reject_unexpected = rospy.get_param('~reject_unexpected', False)
        self.reject_unexpected_srv = rospy.Service('~reject_unexpected', SetBool, self.set_reject_unexpected)

        self.reject_unaligned = rospy.get_param('~reject_unaligned', False)
        self.reject_unaligned_srv = rospy.Service('~reject_unaligned', SetBool, self.set_reject_unaligned)

        self.align_heads = rospy.get_param('~align_heads', False)
        self.align_heads_srv = rospy.Service('~align_heads', SetBool, self.set_align_heads)

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
            self.fit()
        else:
            rospy.loginfo("set fitter inactive")

        self.publish_strings()
        return {'success': True, 'message' : '' }

    def set_exclude_short_strings(self, req):
        if req.data != self.exclude_short_strings:
            self.exclude_short_strings = req.data
            rospy.loginfo(f"set exclude_short_strings to {self.exclude_short_strings}")
            self.publish_strings()
        return {'success': True, 'message' : '' }

    def set_reject_unexpected(self, req):
        if req.data != self.reject_unexpected:
            self.reject_unexpected = req.data
            rospy.loginfo(f"set reject_unexpected to {self.reject_unexpected}")
            self.publish_strings()
        return {'success': True, 'message' : '' }

    def set_reject_unaligned(self, req):
        if req.data != self.reject_unaligned:
            self.reject_unaligned = req.data
            rospy.loginfo(f"set reject_unaligned to {self.reject_unaligned}")
            self.publish_strings()
        return {'success': True, 'message' : '' }

    def set_align_heads(self, req):
        if req.data != self.align_heads:
            self.align_heads = req.data
            rospy.loginfo(f"set align_heads to {self.align_heads}")
            self.publish_strings()
        return {'success': True, 'message' : '' }

    def load_from_file(self, _req = None):
        with open(self.storage_path, 'r') as f:
            self.active = False
            plain_strings = yaml.safe_load(f)
            self.strings = [
                String(key= str(s['key']),
                       head= np.array(s['head']),
                       bridge= np.array(s['bridge']))
                       for s in plain_strings]
            rospy.loginfo(f"loaded {len(self.strings)} strings from file {self.storage_path}. Set inactive to avoid overwriting them.")
            self.publish_strings()
        return {}

    def store_to_file(self, req):
        with open(self.storage_path, 'w') as f:
            yaml.dump([s.as_plain_types for s in self.strings], f, sort_keys= False)
            rospy.loginfo(f"stored strings to file {self.storage_path}")
        return {}

    def onsets_cb(self, msg):
        rospy.loginfo(f"got {len(msg.markers)} onsets to fit")
        if not self.active:
            return

        self.onsets = {}
        for m in msg.markers:
            if m.ns != "unknown" and m.ns != "":
                self.onsets[m.ns] = \
                    self.onsets.get(m.ns, []) + [m.pose.position]
        self.fit()
        self.publish_strings()

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

    def split_unexpected_strings(self, strings : List[String]):
        expected_keys = [f"{k}{o}" for o in [2,3,4,5] for k in ["d", "e", "fis", "a", "b"]]+["d6"]
        expected_strings = [s for s in strings if s.key in expected_keys]
        unexpected_strings = [s for s in strings if s.key not in expected_keys]
        if len(unexpected_strings) > 0:
            rospy.logwarn(f"drop unexpected strings {[s.key for s in unexpected_strings]}")
        return (expected_strings, unexpected_strings)

    def split_unaligned_strings(self, strings : List[String]):
        s_threshold = 2.0
        directions = self.project(np.array([s.direction for s in strings]))
        angles = np.arctan2(directions[:,0], directions[:,1])
        # Modified Z-score, e.g.,
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        def score(data):
            d= np.median(data)-data
            mad= np.median(np.abs(d))
            return 0.6745*d/mad
        scores = np.abs(score(angles))
        keys= np.array([s.key for s in strings])

        strings_aligned = np.array([s for s in strings], dtype=object)[scores < s_threshold].tolist()
        strings_unaligned = np.array([s for s in strings], dtype=object)[scores >= s_threshold].tolist()

        if len(strings_unaligned) > 0:
            rospy.loginfo(
                f"found outliers: {', '.join([s.key for s in strings_unaligned])}\n"
                f"mod-Z-scores: {', '.join([f'{k}({s:2f})' for (k, s) in sorted(zip(keys, scores), key=lambda x: -x[1])])}"
            )

        return (strings_aligned, strings_unaligned)

    def do_align_heads(self, strings : List[String]):
        # copy to adapt below
        aligned_strings = copy.deepcopy(strings)

        heads = np.array([s.head for s in aligned_strings], dtype=float)
        heads2d = self.project(heads)
        model, inliers = ransac(
            heads2d,
            LineModelND,
            min_samples=2,
            residual_threshold=0.01,
            max_trials=1000
            )
        rospy.loginfo(f'straight head fit with {np.sum(inliers)} inliers from {len(aligned_strings)} strings')

        # 2d line model of the straight head
        model_origin = model.params[0]
        model_direction = model.params[1]

        for s in aligned_strings:
            # solve line intersection between ransaced straight head and each string in the 2d projection
            _, t = np.linalg.solve(
                np.array([
                    self.project(s.direction),
                    -model_direction], dtype=float).T,
                model_origin - self.project(s.head))
            intersect = model_origin + t * model_direction

            # adapt head position along string in 3d
            length_offset = (intersect - self.project(s.head)) @ self.project(s.direction)
            s.head += s.direction * length_offset

        return aligned_strings

    def fit(self):
        strings = []

        log_info = ""
        for k in sorted(self.onsets.keys()):
            if len(self.onsets[k]) >= 2:
                pts = np.array(
                    [(p.x, p.y, p.z) for p in self.onsets[k]],
                    dtype=float)

                inlier_threshold= 0.01

                # # special case because the library is broken
                # # https://github.com/scikit-image/scikit-image/pull/6755
                # if pts.shape[0] == 2:
                #     model = LineModelND()
                #     model.estimate(pts)
                #     inliers = model.residuals(pts) < inlier_threshold
                # else:
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

                # ensure string direction to face left and/or upward
                if direction @ np.array([1, 1, 1]) < 0:
                    direction = -direction

                inlier_positions_on_string = (inlier_pts - model.params[0]) @ direction
                min_pos_on_string = np.min(inlier_positions_on_string)
                min_pt = origin + min_pos_on_string * direction

                max_pos_on_string = np.max(inlier_positions_on_string)
                max_pt = model.params[0] + max_pos_on_string * direction

                length = max_pos_on_string-min_pos_on_string

                strings.append(
                    String(
                        key= note_to_string(k),
                        head= min_pt,
                        bridge= max_pt,
                    )
                )

        if len(log_info) > 0:
            rospy.loginfo(log_info)

        self.strings = strings

    def publish_strings(self):

        strings = self.strings
        markers = MarkerArray(markers= [Marker(action = Marker.DELETEALL)])

        if self.exclude_short_strings:
            short_strings = [s for s in strings if s.length < 0.05]
            strings = [s for s in strings if s.length >= 0.05]
            short_strings_markers = [sm for s in short_strings for sm in s.markers]
            for m in short_strings_markers:
                m.color = ColorRGBA(0.6,0.6,0.6, 0.3)
                m.ns = "short "+m.ns
            markers.markers.extend(short_strings_markers)

        if self.reject_unexpected:
            strings, unexpected_strings = self.split_unexpected_strings(strings)
            unexpected_strings_markers= [sm for s in unexpected_strings for sm in s.markers]
            for m in unexpected_strings_markers:
                m.color = ColorRGBA(0.6,0.6,0.6, 0.3)
                m.ns = "unexpected "+m.ns
            markers.markers.extend(unexpected_strings_markers)

        if self.reject_unaligned:
            strings, unaligned_strings = self.split_unaligned_strings(strings)
            unaligned_strings_markers = [sm for s in unaligned_strings for sm in s.markers]
            for m in unaligned_strings_markers:
                m.color = ColorRGBA(0.8,0.1,0.1, 0.5)
                m.ns = "unaligned "+m.ns
            markers.markers.extend(unaligned_strings_markers)

        if self.align_heads and len(strings) > 1:
            cut_strings = self.do_align_heads(strings)
            cut_strings_markers = [sm for s in cut_strings for sm in s.markers]
            for m in cut_strings_markers:
                m.scale.x = m.scale.y = m.scale.x/2
                m.ns = "aligned heads " + m.ns
            markers.markers.extend(cut_strings_markers)

        self.pub_head_poses.publish(PoseArray(
            header= Header(stamp= rospy.Time.now(), frame_id= "base_footprint"),
            poses= [s.head_pose() for s in strings]
            ))

        markers.markers.extend([sm for s in strings for sm in s.markers])

        self.pub_string_markers.publish(markers)
        self.pub_strings.publish(ChordophoneEstimation(
            header= Header(stamp= rospy.Time.now(), frame_id= "base_footprint"),
            strings= [s.as_msg for s in strings])
            )

def main():
    rospy.init_node('string_fitter')
    sf = StringFitter()
    sf.start()
    rospy.spin()


if __name__ == '__main__':
    main()
