#!/bin/bash

# optional argument allows for custom filename

rosbag record \
  --tcpnodelay \
  -o "guzheng${1:+_$1}" \
  \
  /episode/state \
  /episode/action_parameters \
  \
  /joint_states \
  /tf \
  /tf_static \
  \
  /pluck/execute_path/goal \
  /pluck/execute_path/result \
  /pluck/trajectory \
  /pluck/executed_trajectory \
  /pluck/planned_path \
  /pluck/executed_path \
  \
  /guzheng/audio \
  /guzheng/audio_info \
  /guzheng/onsets \
  \
  /hand/rh/tactile \
  /guzheng/plucks \
  \
  /diagnostics_agg \
  /mannequin_mode_active \
  \
  /move_group/monitored_planning_scene \
  /execute_trajectory/goal \
  /execute_trajectory/result \
