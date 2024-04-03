# Demo flow

### Startup the framework

- launch and calibrate regular PR2
- move Guzheng in front of PR2, connect microphone through USB soundcard to basestation
- `mountpr2.sh` on basestation to share workspace between c1 and basestation
- launch `all.launch` (which launches nodes on both basestation and c1)
- launch `rviz.launch` on basestation for a pre-setup visualization

- run `rosrun tams_pr2_guzheng cli` somewhere and look through `help` for available commands.
  It provides a command line with various commands to control the framework.
  Everything can be done outside the CLI as well with higher granularity, but it's a useful entry point.

### Teach-in initial guesses for string position

- move MoveIt joint model group `manipulation` to `guzheng_initial` (e.g., through `goto_guzheng_initial.sh` on basestation)
- activate mannequin mode (optionally set the head back to default position controller via `hold_head_still.sh` on basestation)
- teach-in string plucking until all strings appear roughly in the right spot
- disable mannequin mode & go back to `guzheng_initial`

### Geometry exploration

throughout geometry exploration, optionally run `rqt_reconfigure` on `fingertips`, `guzheng`, and `plectrum_poses`
to adjust detection thresholds, timing, Cartesian plectrum poses, and string reconstruction options as dynamic calibration steps.

- explore geometry of demonstrated strings, e.g.,
  `roslaunch tams_pr2_guzheng explore.launch strategy:=geometry string:=all`
  (there are various parameters)
- notice that you have to confirm trajectory execution at first in the RvizVisualToolsGui as breakpoints are added before actual execution.
  Confirming with `continue` will drop further questions.
- disable string fitter once you are happy with the current result (dynamic reconfigure `active` flag)
  optionally store current geometry (`guzheng/string_fitter/store_to_file` service)

### Dynamics exploration

- `roslaunch tams_pr2_guzheng explore.launch strategy:=avpe string:=all direction:=0.0`
  (or pick strings as space-separated list and direction from {-1.0, 1.0} to be more selective)

### Reproduction

- Make the gathered plucks available for playing
  `mv $(rospack find tams_pr2_guzheng)/data/plucks_explore.json $(rospack find tams_pr2_guzheng)/data/plucks.json`

- start the module that receives onset lists (`music_perception/Piece`) and builds plucking paths
  `rosrun tams_pr2_guzheng play_piece.py`

- E.g., run repeat after me demo node that listens to note onsets and will try to imitate melodies
  `rosrun tams_pr2_guzheng repeat_after_me.py`

# Information Structure

## Raw Inputs

### Native PR2

`/joint_states`    - current joint readings for PR2 & Shadow Hand
`/hand/rh/tactile` - BioTac readings
`/tf`              - Transforms
`/tf_static`
`/diagnostics_agg`       - Diagnostics system (useful to detect runtime faults)
`/mannequin_mode_active` - Is mannequin mode active (and the robot cannot move by itself)?

(tf already includes plectrum/fingertip positions and detected string frames)

### Guzheng

`/guzheng/audio`         - unused
`/guzheng/audio_stamped` - time-stamped audio
                         depending on the publisher audio is ros::Time audio or audio pipeline time (drifts over time)
`/guzheng/audio_info`    - meta data (1 constant latched message)

### MoveIt

`/move_group/monitored_planning_scene` - MoveIt's world model
`/execute_trajectory/goal`             - MoveIt's Trajectory Execution action (which splits trajectories for hand/arm controller and sends them on)
`/execute_trajectory/result`

## Experiment control flow

`/run_episode/goal`          - generate, execute, and analyze a single pluck (including approach motion)
`/run_episode/result`

`/episode/state`             - "start"/"end" before/after path is sent to /pluck/pluck action
`/episode/action_parameters` - selected and executed parameters for single episode pluck

`/pluck/execute_path/goal`   - generate and execute a generic Cartesian trajectory with a target frame
`/pluck/execute_path/result`

`/pluck/pluck/goal`          - same as `execute_path` but provides the following additional debugging output/data collection
`/pluck/pluck/result`
`/pluck/commanded_path`      - Cartesian path to execute in pluck action
`/pluck/planned_path`        - path from generated joint trajectory
`/pluck/executed_path`       - eventually executed path
`/pluck/projected_img`       - image summarizing the three paths in 2d string space
`/pluck/trajectory`          - generated Trajectory
`/pluck/executed_trajectory` - recorded trajectory execution
`/pluck/active_finger`       - current finger used in /pluck action (used for projection)
`/pluck/keypoint`            - keypoint of the ruckig parameterization selected in `run_episode`

`/fingertips/plucks`                   - detected plucking events
`/fingertips/plucks_projected`         - all projected plucks
`/fingertips/plucks_latest`            - latest projected pluck only for visualization
`/fingertips/pluck_detector/signal`    - thresholding signal
`/fingertips/pluck_detector/detection` - high/low signal to debug signal processing
`/fingertips/pluck_detector/parameter_descriptions` - dynamic reconfigure for threshold
`/fingertips/pluck_detector/parameter_updates`
`/fingertips/pluck_projector/parameter_descriptions` - dynamic reconfigure for pluck projection
`/fingertips/pluck_projector/parameter_updates`

`/guzheng/onsets`                      - currently detected NoteOnsets
`/guzheng/onsets_markers`              - Markers generated from onsets
`/guzheng/onsets_projected`            - all onsets projected according to current parameters
`/guzheng/onsets_latest`               - latest onsets projected for visualization
`/guzheng/cqt`                         - cqt generated as a side-product by `detect_onset`
`/guzheng/onset_detector/envelope`     - envelope used to extract peaks as maxima
`/guzheng/onset_detector/compute_time` - debugging topic to measure computation time
`/guzheng/onset_detector/drift`        - drift compensation for audio input (onsets are shifted by the value)
`/guzheng/spectrogram`                 - image visualization of cqt
`/guzheng/onset_projector/parameter_descriptions` - dynamic reconfigure for onset projection
`/guzheng/onset_projector/parameter_updates`

`/guzheng/events`    - unused (alternative projector input)
`/fingertips/events` - unused (alternative projector input)

`/guzheng/fitted_strings` - string markers for all strings currently fitted through projected note onsets
                          or loaded strings from file

## TF frames

`target_pluck_string` - a dynamic frame published when `run_episode` attempts to target a string
`rh_{finger}_plectrum` - tip of the plectrum as manually calibrated
