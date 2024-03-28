#!/usr/bin/env python

import rospy
import tf2_ros

from tams_pr2_guzheng.msg import ChordophoneEstimation
from tams_pr2_guzheng.utils import String
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

# from moveit_commander import PlanningSceneInterface
# from moveit_msgs.msg import CollisionObject
# from shape_msgs.msg import SolidPrimitive
# from geometry_msgs.msg import Pose


    # def publish_to_planning_scene(self, strings):
    #     if self.planning_scene_interface is None:
    #         self.planning_scene_interface = PlanningSceneInterface()
    #     cos = []
    #     for s in strings:
    #         co = CollisionObject()
    #         co.id = s["key"]
    #         co.header.frame_id = "guzheng/"+s["key"]+"/head"
    #         co.operation = CollisionObject.ADD
    #         sp = SolidPrimitive()
    #         sp.type = SolidPrimitive.CYLINDER
    #         sp.dimensions = [0.003, s["length"]]
    #         co.primitives.append(sp)
    #         pose = Pose()
    #         pose.orientation.w = 1.0
    #         pose.position.z = s["length"]/2
    #         co.primitive_poses.append(pose)
    #         cos.append(co)
    #     self.planning_scene_interface.applyCollisionObjects(cos)

class ChordophoneStatePropagator:
    def __init__(self):
        self.tf_broadcast = tf2_ros.StaticTransformBroadcaster()
        self.pub_markers = rospy.Publisher(
            'estimate_markers',
            MarkerArray,
            queue_size=1,
            latch=True,
            tcp_nodelay=True
            )

        self.sub_strings = rospy.Subscriber(
            'estimate',
            ChordophoneEstimation,
            self.estimation_cb,
            tcp_nodelay=True,
            queue_size=1
            )

        # self.psi = PlanningSceneInterface()

    def estimation_cb(self, msg):
        strings = [String.from_msg(s) for s in msg.strings]
        # TODO: respect "up"

        markers = [m for s in strings for m in s.markers]
        self.pub_markers.publish(MarkerArray(markers=markers))

        # crude hack. WTF
        # TODO: implement sendTransform*s* in StaticBroadcaster
        tf_msg = TFMessage(
            transforms=[t for s in strings for t in s.tfs])
        self.tf_broadcast.pub_tf.publish(tf_msg)


def main():
    rospy.init_node('chorodophone_state_propagator')
    ChordophoneStatePropagator()
    rospy.spin()


if __name__ == '__main__':
    main()
