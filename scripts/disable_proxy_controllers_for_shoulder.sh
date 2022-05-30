#!/bin/bash

rosservice call /pr2_controller_manager/switch_controller "stop_controllers:
- 'r_arm_controller'
strictness: 2"

rosservice call /pr2_controller_manager/unload_controller "name: 'r_arm_controller'"
rosparam delete /r_arm_controller/gains/r_shoulder_lift_joint/proxy
rosparam delete /r_arm_controller/gains/r_shoulder_pan_joint/proxy
rosservice call /pr2_controller_manager/load_controller "name: 'r_arm_controller'"

rosservice call /pr2_controller_manager/switch_controller "start_controllers:
- 'r_arm_controller'
strictness: 2"
