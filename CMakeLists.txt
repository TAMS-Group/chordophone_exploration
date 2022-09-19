cmake_minimum_required(VERSION 3.0.2)
project(tams_pr2_guzheng)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  audio_common_msgs
  bio_ik
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  nav_msgs
  rviz_visual_tools
  sensor_msgs
  sr_robot_msgs
  std_msgs
  tf2_ros
  trajectory_msgs
)

find_package(backward_ros QUIET)

generate_dynamic_reconfigure_options(
  cfg/Offsets.cfg
  cfg/Threshold.cfg
)

add_action_files(
DIRECTORY
  action
FILES
  ExecutePath.action
)

add_message_files(
DIRECTORY
  msg
FILES
  ActionParameters.msg
  BiotacStamped.msg
  CQTStamped.msg
  EpisodeState.msg
  NoteOnset.msg
  PluckEpisodeV1.msg
  RunEpisodeRequest.msg
)

generate_messages(
DEPENDENCIES
  actionlib_msgs
  audio_common_msgs
  geometry_msgs
  moveit_msgs
  nav_msgs
  sr_robot_msgs
  trajectory_msgs
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# this package consists of one-file examples only
function(one_filer name)
    add_executable(${PROJECT_NAME}_${name} src/${name}.cpp)
    add_dependencies(${PROJECT_NAME}_${name} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    target_link_libraries(${PROJECT_NAME}_${name} ${catkin_LIBRARIES})
    set_target_properties(${PROJECT_NAME}_${name} PROPERTIES OUTPUT_NAME ${name} PREFIX "")
    install(TARGETS ${PROJECT_NAME}_${name}
      RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endfunction()

one_filer(alternate_poses)
one_filer(detect_pluck)
one_filer(stop_on_contact)
one_filer(pluck_from_path)
one_filer(feel_string)
