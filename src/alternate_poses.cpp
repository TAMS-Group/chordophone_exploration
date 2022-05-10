#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit_msgs/RobotTrajectory.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "detect_strings");
  ros::AsyncSpinner spinner{ 2 };
  spinner.start();
  ros::NodeHandle nh{"~"};

  moveit::planning_interface::MoveGroupInterface mgi{ "right_arm" };
  mgi.setGoalTolerance(0.0);

  double factor{ nh.param("factor", 0.1) };
  mgi.setMaxVelocityScalingFactor(factor);
  mgi.setMaxAccelerationScalingFactor(factor);

//  mgi.rememberJointValues("extended", {0.01400007014833582, -0.1503130120423125, -0.10460533795017679, -0.8218330893876233, -1.5417190468481798});
//  mgi.rememberJointValues("side", {-1.4732617386197226, 0.8506141181302039, 0.0538252305033029, -2.1131878349293762, 0.03634948516335991});

  mgi.rememberJointValues("back", {-1.1988408351678985, 1.0235261409558147, -0.9605794011128038, -1.8188689786865082, -1.1118573437184969});
  mgi.rememberJointValues("front", {-1.0987725480179433, 0.9397771670823535, -0.8969185350439461, -1.7509715150475975, -1.0414574388996534});

//  while(ros::ok())
    for(auto&& target : {"back", "front"}){
      ROS_INFO_STREAM("planning path to " << target);

      mgi.setNamedTarget(target);

      auto execution_result{ mgi.move() };
      ROS_INFO_STREAM("execution state after move: " << execution_result);

      ros::Duration(5.0).sleep();
    }

  return 0;
}
