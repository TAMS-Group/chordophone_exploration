cmake_minimum_required(VERSION 3.0.2)
project(tams_pr2_guzheng)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  bio_ik
  moveit_ros_planning_interface
  rviz_visual_tools
  sr_robot_msgs
  std_msgs
  tf2_ros
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

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
one_filer(stop_on_contact)
one_filer(path_to_traj)
