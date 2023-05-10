#!/bin/bash

# TODO: fix used topics / list them here
# here is a complete list of all currently-available topics related to this:
# $ rostopic list | grep pluck
# /guzheng/pluck_detector/detection
# /guzheng/pluck_detector/parameter_descriptions
# /guzheng/pluck_detector/parameter_updates
# /guzheng/pluck_detector/signal
# /guzheng/pluck_projector/parameter_descriptions
# /guzheng/pluck_projector/parameter_updates
# /guzheng/plucks
# /guzheng/plucks_projected
# /pluck/active_finger
# /pluck/commanded_path
# /pluck/execute_path/cancel
# /pluck/execute_path/feedback
# /pluck/execute_path/goal
# /pluck/execute_path/result
# /pluck/execute_path/status
# /pluck/executed_path
# /pluck/executed_trajectory
# /pluck/keypoint
# /pluck/keypoint_array
# /pluck/planned_path
# /pluck/pluck/cancel
# /pluck/pluck/feedback
# /pluck/pluck/goal
# /pluck/pluck/result
# /pluck/pluck/status
# /pluck/projected_img
# /pluck/trajectory
# /pluck_from_path/planning_scene_monitor/parameter_descriptions
# /pluck_from_path/planning_scene_monitor/parameter_updates
# $ rostopic list | grep guzheng
# /guzheng/active_finger
# /guzheng/audio
# /guzheng/audio_info
# /guzheng/audio_stamped
# /guzheng/cqt
# /guzheng/events
# /guzheng/fitted_strings
# /guzheng/latest_events
# /guzheng/onset_detector/compute_time
# /guzheng/onset_detector/drift
# /guzheng/onset_detector/envelope
# /guzheng/onset_projector/parameter_descriptions
# /guzheng/onset_projector/parameter_updates
# /guzheng/onsets
# /guzheng/onsets_latest
# /guzheng/onsets_markers
# /guzheng/onsets_projected
# /guzheng/pluck_detector/detection
# /guzheng/pluck_detector/parameter_descriptions
# /guzheng/pluck_detector/parameter_updates
# /guzheng/pluck_detector/signal
# /guzheng/pluck_projector/parameter_descriptions
# /guzheng/pluck_projector/parameter_updates
# /guzheng/plucks
# /guzheng/plucks_projected
# /guzheng/spectrogram
# $ rostopic list | grep episode
# /episode/action_parameters
# /episode/state
# /run_episode/cancel
# /run_episode/feedback
# /run_episode/goal
# /run_episode/result
# /run_episode/status


# optional argument allows for custom filename

rosbag record \
  --tcpnodelay \
  -o "guzheng${1:+_$1}" \
  \
  /joint_states \
  /tf \
  /tf_static \
  /hand/rh/tactile \
  /diagnostics_agg \
  /mannequin_mode_active \
  \
  /move_group/monitored_planning_scene \
  /execute_trajectory/goal \
  /execute_trajectory/result \
  \
  /run_episode/goal \
  /run_episode/result \
  /episode/action_parameters \
  /episode/state \
  \
  /pluck/pluck/goal \
  /pluck/pluck/result \
  /pluck/execute_path/goal \
  /pluck/execute_path/result \
  /pluck/trajectory \
  /pluck/executed_trajectory \
  /pluck/planned_path \
  /pluck/executed_path \
  \
  /guzheng/audio_stamped \
  /guzheng/audio_info \
  /guzheng/cqt \
  /guzheng/onsets \
  /guzheng/onsets_markers \
  /guzheng/onsets_latest \
  /guzheng/onset_detector/compute_time \
  \
  /guzheng/plucks \
  \
  /guzheng/onset_projector/parameter_updates \
  /guzheng/pluck_projector/parameter_updates \
  \
  \
  /guzheng/onsets_projected \
  /guzheng/fitted_strings \

