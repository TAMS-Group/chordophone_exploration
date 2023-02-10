#!/usr/bin/env python

import rospy

import tf2_py as tf2
import tf2_ros
import tf2_geometry_msgs

from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from tams_pr2_guzheng.cfg import TimeOffsetConfig

from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

import time
from copy import deepcopy

FINGERS = ["th", "ff", "mf", "rf"]

class LoopDetector:
    def __init__(self):
        self.time= rospy.Time.now()
    def has_looped(self):
        last_call= self.time
        self.time= rospy.Time.now()
        return last_call > self.time
    def __bool__(self):
        return self.has_looped()

class TransformListenerCb(tf2_ros.TransformListener):
    def __init__(self, buffer, cb):
        self.cb = cb
        super().__init__(buffer)
    def static_callback(self, data):
        super().static_callback(data)
        # we only need callbacks for plectrum msgs
        if len(data.transforms) > 0 and data.transforms[0].child_frame_id.endswith('_plectrum'):
            self.cb(data)

class Projector:
    def __init__(self):
        self.tf_buffer= tf2_ros.Buffer(rospy.Duration(30.0), debug= False)

        self.loop= LoopDetector()

        self.marker_scale= rospy.get_param("~marker_scale", 1.0)
        self.base_frame= rospy.get_param("~base_frame", "base_footprint")
        self.default_finger= rospy.get_param("~default_finger", "ff")

        self.reset()

        self.pub= rospy.Publisher('events_projected', MarkerArray, queue_size= 1, latch= True, tcp_nodelay= True)

        # server directly sets config correctly
        self.config= None
        self.dr_server= DynamicReconfigureServer(TimeOffsetConfig, self.offset_cb)

        self.tf_listener= TransformListenerCb(self.tf_buffer, lambda tfs: self.publish())

        self.sub_finger= rospy.Subscriber('active_finger', String, self.finger_cb, queue_size= 100, tcp_nodelay= True)
        self.sub_events= rospy.Subscriber('events', Header, self.event_header_cb, queue_size= 100, tcp_nodelay= True)
        self.sub_marker= rospy.Subscriber('events_markers', MarkerArray, self.event_marker_array_cb, queue_size= 100, tcp_nodelay= True)


    def reset(self):
        self.id= 0
        self.events= []
        self.last_publish= time.time()
        self.tf_buffer.clear()
        self.finger= self.default_finger

    def finger_cb(self, finger):
        if finger.data not in FINGERS:
            rospy.logerr(f"'{finger.data}' is not a valid finger. Ignoring.")
        else:
            self.finger = finger.data

    def offset_cb(self, config, level):
        self.config= config
        self.publish()
        return config

    def event_header_cb(self, msg):
        m= Marker()
        m.type= Marker.SPHERE
        m.action= Marker.ADD
        m.ns= "event"
        m.color.a= 1.0
        m.color.r= 0.8
        m.color.g= 0.0
        m.color.b= 0.8
        m.scale.x= 0.01
        m.scale.y= 0.01
        m.scale.z= 0.01

        m.header= msg
        #m.header.frame_id= self.config.frame
        m.pose.orientation.w= 1.0
        m.pose.position.y= 0.01
        m.pose.position.z= 0.025

        self.event_marker_cb(m)

    def event_marker_array_cb(self, msg):
        for m in msg.markers:
            self.event_marker_cb(m)

    def event_marker_cb(self, marker):
        if self.loop:
            rospy.loginfo("detected loop")
            self.reset()
        f= self.finger
        marker.header.frame_id = f'rh_{f}_plectrum'
        buffer_target_frame = f'rh_{f}_biotac_link'
        if not self.tf_buffer.can_transform(self.base_frame, buffer_target_frame, marker.header.stamp+rospy.Duration(TimeOffsetConfig.max['delta_t']), timeout= rospy.Duration(TimeOffsetConfig.max['delta_t']+0.2)):
            rospy.logwarn("throw away event because transform is not available")
            return
        # buffer local trajectory of plectrum to pick precise value during calibration
        buffer= tf2_ros.Buffer(cache_time= rospy.Duration(1 << 30), debug= False)  # inf is not supported... :(
        buffer.set_transform_static(self.tf_buffer.lookup_transform(buffer_target_frame, marker.header.frame_id, rospy.Time()), '')
        for dt in np.arange(TimeOffsetConfig.min['delta_t'], TimeOffsetConfig.max['delta_t']+0.01, 0.01):
            try:
                buffer.set_transform(self.tf_buffer.lookup_transform(self.base_frame, buffer_target_frame, marker.header.stamp+rospy.Duration(dt)), '')
            except Exception as e:
                rospy.logwarn('throw away event because transform in temporal vicinity is not available (delta "{}")'.format(dt))
                rospy.logwarn(e)
                return

        # ensure on the projector side that ids cannot collide
        # the namespace can be defined by the sender though
        marker.id= self.id
        self.id= self.id+1
        marker.scale.x *= self.marker_scale
        marker.scale.y *= self.marker_scale
        marker.scale.z *= self.marker_scale

        self.events.append((marker, f, buffer))
        self.publish()

    def publish(self):
        now= time.time()
        if True or now - self.last_publish > 1.0:
            self.last_publish= now

            markers= MarkerArray()
            for marker, finger, buffer in self.events:
                buffer.set_transform_static(self.tf_buffer.lookup_transform(f'rh_{finger}_biotac_link', marker.header.frame_id, rospy.Time()), '')
                p= PoseStamped(header= deepcopy(marker.header))
                p.pose.position= deepcopy(marker.pose.position)
                p.pose.orientation.w= 1.0

                p.header.stamp+= rospy.Duration(self.config.delta_t)

                try:
                    p= buffer.transform(p, self.base_frame)
                except Exception as e:
                    rospy.logerr('transform in publish failed:\n'+str(e)+"\nonly know these: "+buffer.all_frames_as_yaml())
                    continue
                m= deepcopy(marker)
                m.header= p.header
                m.pose= p.pose
                markers.markers.append( m )
            self.pub.publish(markers)


def main():
    rospy.init_node('project_events_to_frame')
    Projector()
    rospy.spin()

if __name__ == '__main__':
    main()
