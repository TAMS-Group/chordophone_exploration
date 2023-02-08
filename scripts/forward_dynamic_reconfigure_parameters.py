#!/usr/bin/env python

import rospy
from dynamic_reconfigure.srv import Reconfigure
from dynamic_reconfigure.msg import Config

follower= None
def config_cb(msg):
    global follower
    set_values= False 
    # HACK: I only use this for this one use-case anyway...
    try:
        idx = next(idx for idx,e in enumerate(msg.doubles) if e.name == 'delta_t')
        msg.doubles.pop(idx)
    except Exception:
        pass
    while not rospy.is_shutdown() and not set_values:
        try:
            follower(msg)
            set_values= True
        except Exception as e:
            rospy.logwarn(e)

def main():
    global follower
    rospy.init_node('forward_dynamic_reconfigure')
    follower = rospy.ServiceProxy(rospy.resolve_name('follower')+'/set_parameters', Reconfigure, persistent= True)
    follower.wait_for_service()
    rospy.Subscriber(rospy.resolve_name('source')+'/parameter_updates', Config, config_cb, queue_size= 10)
    rospy.spin()


if __name__ == '__main__':
    main()
