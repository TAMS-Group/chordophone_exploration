#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node('publish_path')
    pub = rospy.Publisher('pluck/path', Path, latch=True, queue_size=1)

    path = Path()
    path.header.frame_id = 'guzheng/d6/head'

    waypoints = [
        (.05, 0.00 +0.000, 0.01+0.015),
        (.05,-0.015+0.000, 0.00+0.015),
        (.05,-0.020+0.000, 0.00+0.015),
        (.05,-0.025+0.000, 0.02+0.015),
        (.05, 0.00 +0.000, 0.01+0.015)
        ]

    for x, y, z in waypoints:
        p = PoseStamped()
        p.pose.position.x = x  # 0 - 0.15
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation.w = 1.0
        path.poses.append(p)
    rospy.sleep(1.0)
    pub.publish(path)
    rospy.sleep(1.0)
#    rospy.spin()


if __name__ == "__main__":
    main()
