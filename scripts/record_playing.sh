#!/bin/bash

# optional argument allows for custom filename

rosbag record \
  -o "guzheng${1:+_$1}" \
  \
  /guzheng/audio \
  /guzheng/audio_info \
  /joint_states \
  /hand/joint_states_original \
  /hand/rh/tactile \
  /tf \
  /tf_static \
  /diagnostics_agg \
  /mannequin_mode_active \
  /move_group/monitored_planning_scene \
  /move_group/display_planned_path \


