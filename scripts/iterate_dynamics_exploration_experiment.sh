#!/bin/bash

# assumes all relevant strings are localized through onset fitting
# and continuous fitting is disabled
# automated experiment script to explore dynamics for a set of conditions

rostopic pub -1 /moveit_by_name moveit_by_name/Command "group: 'manipulation'
target: 'guzheng_initial'" >&-

TRIALS=250

F=`mktemp`

# d6 -1.0 1

cat > $F <<EOF
d6 1.0 1
d5 1.0 1
d4 1.0 1
EOF

cat $F | while read string direction runs; do
    echo "$string / $direction / $runs"

    if [[ $direction = 1.0 ]]; then
       direction_string=inwards
    else
       direction_string=outwards
    fi
    for i in `seq 1 $runs`; do
       rostopic pub -1 /say std_msgs/String "data: 'exploring string $string $direction_string'" >&-
       roslaunch tams_pr2_guzheng explore.launch strategy:=reduce_variance string:=${string} direction:=${direction} runs:=$TRIALS storage:=${string}_${direction_string}_`date +%Y%m%d` 
    done
done 

rostopic pub -1 /moveit_by_name moveit_by_name/Command "group: 'manipulation'
target: 'guzheng_rest'" >&-
rostopic pub -1 /moveit_by_name moveit_by_name/Command "group: 'head'
target: 'rest'" >&-

rostopic pub -1 /say std_msgs/String "data: 'I am done for the moment. Please let me do something else now.'" >&-
