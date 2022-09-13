#!/usr/bin/env python

import rospy

from visualization_msgs.msg import MarkerArray, Marker
from tams_pr2_guzheng.msg import NoteOnset
from std_msgs.msg import ColorRGBA

import librosa

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class OnsetToMarker:
    def __init__(self):
        self.fmin_note = "C2"
        self.fmin = librosa.note_to_hz(self.fmin_note)
        self.semitones = 84
        self.fmax_note = "C8"
        self.fmax = librosa.note_to_hz(self.fmax_note)

        # self.cmap = plt.get_cmap("gist_rainbow").copy()
        hsv = plt.get_cmap("hsv")
        self.cmap = ListedColormap(
            np.vstack((
                hsv(np.linspace(0, 1, 86)),
                hsv(np.linspace(0, 1, 85)),
                hsv(np.linspace(0, 1, 85)))
                )
            )
        self.cmap.set_bad((0, 0, 0, 1))  # make sure they are visible

    def start(self):
        self.sub_onset = rospy.Subscriber(
            "onsets",
            NoteOnset,
            self.onset_cb,
            queue_size=100,
            tcp_nodelay=True
        )
        self.pub_markers = rospy.Publisher(
            "onsets_markers",
            MarkerArray,
            queue_size=100,
            tcp_nodelay=True
        )

    def color_from_freq(self, freq):
        if freq > 0.0:
            return ColorRGBA(
                *self.cmap(
                    (np.log(freq) - np.log(self.fmin)) /
                    (np.log(self.fmax) - np.log(self.fmin)))
                )
        else:
            return ColorRGBA(*self.cmap.get_bad())

    def onset_cb(self, msg):
        markers = MarkerArray()

        m = Marker()
        if msg.note != '':
            m.ns = msg.note
        else:
            m.ns = "unknown"

        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.header.stamp = msg.header.stamp

        m.scale.x = 0.005
        m.scale.y = 0.005
        m.scale.z = 0.005
        m.color = self.color_from_freq(
            librosa.note_to_hz(msg.note) if msg.note != '' else 0.0
            )
        markers.markers.append(m)
        self.pub_markers.publish(markers)


def main():
    rospy.init_node("onset_to_marker")

    otm = OnsetToMarker()
    otm.start()
    rospy.spin()


if __name__ == "__main__":
    main()
