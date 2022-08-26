#!/usr/bin/env python

# a trivial script to test behavior of plotjuggler with delayed data
# absolutely not related to guzheng playing

import rospy
from sensor_msgs.msg import JointState

rospy.init_node('js_buffer')

last_call = rospy.Time.now()
buffer = []

def cb(msg):
	global buffer

	buffer.append(msg)
	if buffer[0].header.stamp + rospy.Duration(5.0) < rospy.Time.now():
		pub.publish(buffer[0])
		buffer = buffer[1:]
	else:
		rospy.loginfo('{} is still too young compared to {}'.format(buffer[0].header.stamp, rospy.Time.now()))


pub = rospy.Publisher('joint_states_buffered', JointState, queue_size= 100, tcp_nodelay= True)
sub = rospy.Subscriber('joint_states', JointState, cb, queue_size= 100, tcp_nodelay= True)
rospy.spin()
