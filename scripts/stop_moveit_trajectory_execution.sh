#!/bin/sh

rostopic pub -1 /trajectory_execution_event std_msgs/String "data: 'stop'"
