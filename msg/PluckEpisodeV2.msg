## Record a full training episode for plucking a string with all relevant data

int32 id

# stamp defines start time of episode
std_msgs/Header header

time start_execution

# duration of episode
duration length

## Geometry

# robot state before action
moveit_msgs/RobotState start_state

# targeted string (e.g. "c4", "fis3")
string string

# string head frame "guzheng/<string>/head" in planning frame
geometry_msgs/TransformStamped string_head_frame

# plucking finger
string finger # one of th/ff/mf/rf/lf
geometry_msgs/PoseStamped finger_start_pose # in string frame

## Generated Action

tams_pr2_guzheng/ActionParameters action_parameters

## Generated and executed motions

nav_msgs/Path commanded_path

nav_msgs/Path planned_path
trajectory_msgs/JointTrajectory planned_trajectory

nav_msgs/Path executed_path
trajectory_msgs/JointTrajectory executed_trajectory

moveit_msgs/MoveItErrorCodes execution_status

## Audio data

audio_common_msgs/AudioDataStamped audio_data
audio_common_msgs/AudioInfo audio_info

music_perception/CQTStamped cqt

music_perception/NoteOnset[] detected_audio_onsets

## Tactile data

tams_pr2_guzheng/BiotacStamped[] tactile_data

tams_pr2_guzheng/TactilePluck[] detected_tactile_plucks
