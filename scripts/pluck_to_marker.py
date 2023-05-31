#!/usr/bin/env python

import rospy

from visualization_msgs.msg import MarkerArray, Marker
from tams_pr2_guzheng.msg import TactilePluck
from std_msgs.msg import ColorRGBA

import matplotlib.pyplot as plt

class PluckToMarker:
    def __init__(self):
        self.cmap = plt.get_cmap("jet")
        self.cmap.set_bad((0, 0, 0, 1))  # make sure they are visible

        self.min_strength = rospy.get_param("~min_strength", 10)
        self.max_strength = rospy.get_param("~max_strength", 500)

        self.id= 0

    def start(self):
        self.sub_onset = rospy.Subscriber(
            "plucks",
            TactilePluck,
            self.pluck_cb,
            queue_size=100,
            tcp_nodelay=True
        )
        self.pub_markers = rospy.Publisher(
            "plucks_markers",
            MarkerArray,
            queue_size=100,
            tcp_nodelay=True
        )

    def pluck_cb(self, msg):
        markers = MarkerArray()

        m = Marker()

        m.ns = msg.finger
        m.id = self.id
        self.id+= 1
        m.action = Marker.ADD

        m.header = msg.header
        
        if msg.finger not in {"ff", "mf", "rf", "lf", "th"}:
            # stub value if none is set for onset
            m.header.frame_id = "pluck_frame"
        else:
            m.header.frame_id = f"rh_{msg.finger}_plectrum"

        m.type = Marker.CUBE

        m.pose.orientation.w = 1.0

        m.scale.x = 0.005
        m.scale.y = m.scale.x
        m.scale.z = m.scale.x

        if msg.strength > 0:
            m.color = ColorRGBA(
                *self.cmap(
                    (msg.strength-self.min_strength)/(self.max_strength-self.min_strength)
                )
            )
        else:
            m.color = ColorRGBA(*self.cmap.get_bad())

        markers.markers.append(m)
        self.pub_markers.publish(markers)


def main():
    rospy.init_node("pluck_to_marker")

    otm = PluckToMarker()
    otm.start()
    rospy.spin()


if __name__ == "__main__":
    main()
