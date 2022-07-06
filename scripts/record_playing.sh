#!/bin/bash

rosbag record \
  -o guzheng \
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


