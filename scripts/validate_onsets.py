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
        self.onset_failed_pub = rospy.Publisher("onsets_failed_to_validate", NoteOnset, queue_size=50, tcp_nodelay=True)
        self.delay_pub = rospy.Publisher("~audio_tactile_delay", Float32Msg, queue_size=1, tcp_nodelay=True)

        self.pluck_sub = rospy.Subscriber("plucks", TactilePluck, self.pluck_cb, tcp_nodelay=True)
        self.onset_sub = rospy.Subscriber("onsets", NoteOnset, self.onset_cb, tcp_nodelay=True)

    def pluck_cb(self, msg):
        # filter recent plucks for outdated ones
        horizon = rospy.Time.now() - rospy.Duration(2.0)
        self.recent_plucks = [p for p in self.recent_plucks if p.header.stamp > horizon]
        self.recent_plucks.append(msg)

    def onset_cb(self, msg):
        for i, p in reversed(list(enumerate(self.recent_plucks))):
            delay= (msg.header.stamp - p.header.stamp).to_sec()
            if abs(delay) < self.tolerance:
                # this pluck is accounted for now / don't explain other onsets through it as well
                self.recent_plucks.pop(i)
                self.delay_pub.publish(Float32Msg(delay))
                # if so, publish the onset
                self.onset_pub.publish(msg)
                return
        # could not find corresponding pluck
        self.onset_failed_pub.publish(msg)

        # publish delay to most recent pluck, if any
        if len(self.recent_plucks) > 0:
            self.delay_pub.publish(Float32Msg((msg.header.stamp - self.recent_plucks[-1].header.stamp).to_sec()))
        else:
            self.delay_pub.publish(Float32Msg(msg.header.stamp.to_sec()))

if __name__ == "__main__":
    rospy.init_node("validate_onsets")
    ValidateOnsets()
    rospy.spin()