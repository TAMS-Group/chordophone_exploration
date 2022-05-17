#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node('publish_path')
    pub = rospy.Publisher('pluck_path', Path, latch=True, queue_size= 1)

    path = Path()
    path.header.frame_id = 'guzheng/d6/bridge'

    waypoints = [
        (.08, 0.03, 0.01),
        (.08, 0.00, 0.00),
        (.08, -0.03, 0.00),
        ]

    for x,y,z in waypoints:
        p = PoseStamped()
        p.pose.position.x = x # 0 - 0.15
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation.w = 1.0
        path.poses.append(p)
    pub.publish(path)
    rospy.spin()


if __name__ == "__main__":
    main()
