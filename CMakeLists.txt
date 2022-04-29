cmake_minimum_required(VERSION 3.0.2)
project(tams_pr2_guzheng)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  moveit_ros_planning_interface)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(detect_strings src/detect_strings.cpp)
add_dependencies(detect_strings ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(detect_strings ${catkin_LIBRARIES})
install(TARGETS detect_strings
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
