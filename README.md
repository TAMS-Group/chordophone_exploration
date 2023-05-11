# Topics

## Raw Inputs

### Native

/joint_states    - current joint readings for PR2 & Shadow Hand
/hand/rh/tactile - BioTac readings
/tf              - Transforms
/tf_static
/diagnostics_agg       - Diagnostics system (useful to detect runtime faults)
/mannequin_mode_active - Is mannequin mode active (and the robot cannot move by itself)?

(tf already includes plectrum/fingertip positions and detected string frames)

/move_group/monitored_planning_scene - MoveIt's world model
/execute_trajectory/goal             - MoveIt's Trajectory Execution action (which splits trajectories for hand/arm controller and sends them on)
/execute_trajectory/result

/run_episode/goal          - generate, execute, and analyze a single pluck (including approach motion)
/run_episode/result

/episode/action_parameters - selected and executed parameters for single episode pluck
/episode/state             - "start"/"end" before/after path is sent to /pluck/pluck action

/pluck/execute_path/goal   - generate and execute a generic Cartesian trajectory with a target frame
/pluck/execute_path/result 

/pluck/pluck/goal          - same as execute_path but provides the following additional debugging output/data collection
/pluck/pluck/result
/pluck/commanded_path      - Cartesian path to execute in pluck action
/pluck/planned_path        - path from generated joint trajectory
/pluck/executed_path       - eventually executed path
/pluck/projected_img       - image summarizing the three paths in 2d string space
/pluck/trajectory          - generated Trajectory
/pluck/executed_trajectory - recorded trajectory execution
/pluck/active_finger       - current finger used in /pluck action (used for projection)

/pluck/keypoint            - keypoint of the ruckig parameterization selected in run_episode

/guzheng/audio         - unused
/guzheng/audio_stamped - time-stamped audio
                         depending on the publisher audio is ros::Time audio or audio pipeline time (drifts over time)
/guzheng/audio_info    - meta data (1 constant latched message)

/guzheng/cqt           - cqt generated as a side-product by detect_onset

/guzheng/events        - unused

/guzheng/fitted_strings - string markers for all strings currently fitted through projected note onsets
                          or loaded strings from file

/guzheng/plucks_latest
/guzheng/onset_detector/compute_time
/guzheng/onset_detector/drift
/guzheng/onset_detector/envelope

/guzheng/onset_projector/parameter_descriptions
/guzheng/onset_projector/parameter_updates
/guzheng/onsets
/guzheng/onsets_latest
/guzheng/onsets_markers
/guzheng/onsets_projected
/guzheng/pluck_detector/detection
/guzheng/pluck_detector/parameter_descriptions
/guzheng/pluck_detector/parameter_updates
/guzheng/pluck_detector/signal
/guzheng/pluck_projector/parameter_descriptions
/guzheng/pluck_projector/parameter_updates
/guzheng/plucks
/guzheng/plucks_projected
/guzheng/spectrogram
