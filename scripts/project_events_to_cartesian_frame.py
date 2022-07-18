#!/usr/bin/env python

import rospy

import tf2_py as tf2
import tf2_ros
import tf2_geometry_msgs

from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from tams_pr2_guzheng.cfg import OffsetsConfig

from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

import time
from copy import deepcopy

class LoopDetector:
	def __init__(self):
		self.time= rospy.Time.now()
	def has_looped(self):
		last_call= self.time
		self.time= rospy.Time.now()
		return last_call > self.time
	def __bool__(self):
		return self.has_looped()

class Projector:
	def __init__(self):
		self.tf_buffer= tf2_ros.Buffer(rospy.Duration(30.0), debug= False)
		self.tf_listener= tf2_ros.TransformListener(self.tf_buffer)

		self.loop= LoopDetector()

		self.reset()

		self.pub= rospy.Publisher('events_projected', MarkerArray, queue_size= 1)

		# server directly sets config correctly
		self.config= None
		self.dr_server= DynamicReconfigureServer(OffsetsConfig, self.offset_cb)

		self.sub= rospy.Subscriber('events', Header, self.event_cb, queue_size= 100)

	def reset(self):
		self.id= 0
		self.events= []
		self.last_publish= time.time()
		self.tf_buffer.clear()

	def offset_cb(self, config, level):
		self.config = config
		self.publish()
		return config

	def event_cb(self, msg):
		if self.loop:
			rospy.loginfo("detected loop")
			self.reset()

		if not self.tf_buffer.can_transform('base_footprint', self.config.frame, msg.stamp+rospy.Duration(0.1), timeout= rospy.Duration(0.2)):
			rospy.logwarn("throw away event because transform is not available")
			return
		# inf is not supported... :(
		buffer= tf2_ros.Buffer(cache_time= rospy.Duration(1 << 30), debug= False)
		for dt in np.arange(OffsetsConfig.min['delta_t'], OffsetsConfig.max['delta_t'], 0.01):
			try:
				buffer.set_transform(self.tf_buffer.lookup_transform('base_footprint', self.config.frame, msg.stamp+rospy.Duration(dt)), '')
			except Exception as e:
				rospy.logwarn('throw away event because transform in temporal vicinity is not available (delta "{}")'.format(dt))
				return

		m= Marker()
		m.type= Marker.SPHERE
		m.action= Marker.ADD
		m.ns = "audio_event"
		m.id= self.id
		self.id= self.id+1
		m.color.a= 1.0
		m.color.r= 0.8
		m.color.g= 0.0
		m.color.b= 0.8
		m.scale.x= 0.01
		m.scale.y= 0.01
		m.scale.z= 0.01

		m.header= msg
		m.header.frame_id= self.config.frame
		m.pose.orientation.w= 1.0
		m.pose.position.y= 0.01
		m.pose.position.z= 0.025

		self.events.append((m, buffer))
		self.publish()

	def publish(self):
		now= time.time()
		if True or now - self.last_publish > 1.0:
			self.last_publish= now

			markers= MarkerArray()
			for [marker, buffer] in self.events:
				p = PoseStamped(header= deepcopy(marker.header), pose= deepcopy(marker.pose))
				p.header.frame_id = self.config.frame
				p.header.stamp+= rospy.Duration(self.config.delta_t)
				p.pose.position.x+= self.config.offset_x
				p.pose.position.y+= self.config.offset_y
				p.pose.position.z+= self.config.offset_z
				try:
					p = buffer.transform(p, "base_footprint")
				except Exception as e:
					rospy.logerr(e)
					continue
				m = deepcopy(marker)
				m.header= p.header
				m.pose= p.pose
				markers.markers.append( m )
			self.pub.publish(markers)

def main():
	rospy.init_node('project_events_to_frame')
	p= Projector()
	rospy.spin()

if __name__ == '__main__':
	main()
