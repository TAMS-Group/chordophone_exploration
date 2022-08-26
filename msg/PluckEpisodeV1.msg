## Record a full training episode for plucking a string with all relevant data

# stamp defines start time of episode
std_msgs/Header header

# duration of episode
duration length


## Geometry

# robot state before action
sensor_msgs/JointState start_state

# targeted string (by western note name)
string string
# string head frame "guzheng/<note>/head" in planning frame
geometry_msgs/TransformStamped string_head_frame

# plucking finger
string finger # one of th/ff/mf/rf/lf
geometry_msgs/PoseStamped finger_start_pose # in string frame


## Generated Action

# free text definition of the meaning of the parameters below
string actionspace_type
# free parameters depending on value
float64[] action_parameters


## Generated and executed motions

nav_msgs/Path commanded_path
nav_msgs/Path planned_path
trajectory_msgs/JointTrajectory planned_trajectory

nav_msgs/Path executed_path
trajectory_msgs/JointTrajectory executed_trajectory
moveit_msgs/MoveItErrorCodes execution_status

## Audio data

audio_common_msgs/AudioInfo audio_info
audio_common_msgs/AudioData audio_data

time[] detected_audio_onsets

string detected_note


## Tactile data

sr_robot_msgs/Biotac[] tactile_data

time[] detected_tactile_snaps
