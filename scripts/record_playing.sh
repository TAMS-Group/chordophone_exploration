#!/bin/bash

# optional argument allows for custom filename

if [[ "$1" = "--help" ]]; then
  echo "usage: $0 [keywords to add to rosbag name]"
  exit 0
fi

rosbag record \
  --tcpnodelay \
  -o "guzheng${1:+_$1}" \
  \
  /joint_states \
  /hand/rh/tactile \
  /tf \
  /tf_static \
  /diagnostics_agg \
  /mannequin_mode_active \
  \
  /guzheng/audio_stamped \
  /guzheng/audio_info \
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
  /pluck/execute_path/goal \
  /pluck/execute_path/result \
  \
  /pluck/pluck/goal \
  /pluck/pluck/result \
  /pluck/commanded_path \
  \
  /pluck/planned_path \
  /pluck/executed_path \
  /pluck/trajectory \
  /pluck/executed_trajectory \
  /pluck/active_finger \
  /pluck/keypoint \
  \
  /fingertips/plucks \
  /fingertips/plucks_latest \
  /fingertips/pluck_detector/signal \
  /fingertips/pluck_detector/detection \
  /fingertips/pluck_detector/parameter_descriptions \
  /fingertips/pluck_detector/parameter_updates \
  /fingertips/pluck_projector/parameter_descriptions \
  /fingertips/pluck_projector/parameter_updates \
  \
  /guzheng/onsets \
  /guzheng/onsets_haptically_validated \
  /guzheng/onsets_failed_to_validate \
  /guzheng/onsets_markers \
  /guzheng/onsets_latest \
  /guzheng/cqt \
  /guzheng/onset_detector/drift \
  /guzheng/onset_detector/compute_time \
  /guzheng/onset_projector/parameter_descriptions \
  /guzheng/onset_projector/parameter_updates \
  /guzheng/validate_onsets/audio_tactile_delay \
  \
  /guzheng/onset_projector/parameter_updates \
  /guzheng/pluck_projector/parameter_updates \
  \
  /guzheng/fitted_strings \
  /piece \
  /piece_midi_loudness \
  __name:=rosbag_recorder
