#!/usr/bin/env python

import rospy
from music_perception.msg import NoteOnset
from tams_pr2_guzheng.msg import TactilePluck

class ValidateOnsets:

    def __init__(self):
        self.recent_plucks = []
        self.onset_pub = rospy.Publisher("onsets_grounded", NoteOnset, queue_size=50, tcp_nodelay=True)

        self.pluck_sub = rospy.Subscriber("plucks", TactilePluck, self.pluck_cb, tcp_nodelay=True)
        self.onset_sub = rospy.Subscriber("onsets", NoteOnset, self.onset_cb, tcp_nodelay=True)

    def pluck_cb(self, msg):
        # filter recent plucks for outdated ones
        self.recent_plucks = [p for p in self.recent_plucks if p.header.stamp + rospy.Duration(2.0) < rospy.Time.now()]
        self.recent_plucks.append(msg)

    def onset_cb(self, msg):
        for p in self.recent_plucks:
            # TODO: this should actually require the pluck *before* the onset, but signal alignment is not perfect
            if abs((p.header.stamp - msg.header.stamp).to_sec()) < 0.3:
                # if so, publish the onset
                self.onset_pub.publish(msg)
                return
            
if __name__ == "__main__":
    rospy.init_node("validate_onsets")
    ValidateOnsets()
    rospy.spin()