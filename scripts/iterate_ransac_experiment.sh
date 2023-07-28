#!/bin/bash

# semi-automated experiment script to initialize a number of string positions via kinesthetic teaching
# and run the geometry fitting, recording the results into a local rosbag

rosrun tams_pr2_guzheng fitter.sh drop ALL

rostopic pub -1 /moveit_by_name moveit_by_name/Command "group: 'manipulation'
target: 'guzheng_initial'" >&-

rosrun tams_pr2_mannequin_mode turn_on >&-

rostopic pub -1 /say std_msgs/String "data: 'Time to initialize string positions and press enter.'" >&-

echo "Press enter to continue."
read

rosrun tams_pr2_mannequin_mode turn_off >&-

# rostopic pub -1 /moveit_by_name moveit_by_name/Command "group: 'manipulation'
# target: 'guzheng_initial'" >&-

let number_of_strings=$(rostopic echo -n1 /guzheng/fitted_strings | grep ns: | wc -l)-1
let runs=number_of_strings*110

trap "rosnode kill /rosbag_recorder" SIGINT
rosrun tams_pr2_guzheng record_playing.sh $1 &
roslaunch tams_pr2_guzheng explore.launch strategy:=geometry attempts_per_string:=10 runs:=$runs
rosnode kill /rosbag_recorder

rostopic pub -1 /moveit_by_name moveit_by_name/Command "group: 'manipulation'
target: 'guzheng_rest'" >&-