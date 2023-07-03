#!/usr/bin/env python

import rospy
from music_perception.msg import NoteOnset
from tams_pr2_guzheng.msg import TactilePluck
from std_msgs.msg import Float32 as Float32Msg

class ValidateOnsets:

    def __init__(self):
        self.recent_plucks = []

        self.tolerance = rospy.get_param("~tolerance", 0.3)

        self.onset_pub = rospy.Publisher("onsets_haptically_validated", NoteOnset, queue_size=50, tcp_nodelay=True)
        self.delay_pub = rospy.Publisher("~audio_tactile_delay", Float32Msg, queue_size=1, tcp_nodelay=True)

        self.pluck_sub = rospy.Subscriber("plucks", TactilePluck, self.pluck_cb, tcp_nodelay=True)
        self.onset_sub = rospy.Subscriber("onsets", NoteOnset, self.onset_cb, tcp_nodelay=True)

    def pluck_cb(self, msg):
        # filter recent plucks for outdated ones
        horizon = rospy.Time.now() - rospy.Duration(2.0)
        self.recent_plucks = [p for p in self.recent_plucks if p.header.stamp > horizon]
        self.recent_plucks.append(msg)

    def onset_cb(self, msg):
        for p in reversed(self.recent_plucks):
            # this should actually require the pluck *before* the onset, but signal alignment is not perfect
            # Also, onsets come in only every 1.5s, so the later plucks will still be here already
            delay= (msg.header.stamp - p.header.stamp).to_sec()
            if abs(delay) < self.tolerance:
                self.delay_pub.publish(Float32Msg(delay))
                # if so, publish the onset
                self.onset_pub.publish(msg)
                return
        if len(self.recent_plucks) > 0:
            # publish rejected match to most current pluck
            self.delay_pub.publish(Float32Msg(msg.header.stamp - self.recent_plucks[-1].header.stamp).to_sec())

        self.delay_pub.publish(Float32Msg(msg.header.stamp.to_sec()))
            
if __name__ == "__main__":
    rospy.init_node("validate_onsets")
    ValidateOnsets()
    rospy.spin()