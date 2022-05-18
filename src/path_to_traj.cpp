#include <ros/ros.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include <nav_msgs/Path.h>
#include <moveit_msgs/RobotTrajectory.h>

int main(int argc, char** argv){
   ros::init(argc, argv, "path_to_traj");
   ros::NodeHandle nh, pnh{"~"};

   ros::AsyncSpinner spinner{ 1 };
   spinner.start();

   planning_scene_monitor::PlanningSceneMonitor psm{ "robot_description" };
   if(!psm.requestPlanningSceneState()){
      ROS_FATAL_STREAM("failed to get current scene from move_group");
      return 1;
   }
   ROS_INFO("finished loading scene");

   nav_msgs::PathConstPtr path{ ros::topic::waitForMessage<nav_msgs::Path>("pluck/path", nh) };
   ROS_INFO_STREAM("got path with " << path->poses.size() << " poses");

   std::string group_name;
   pnh.param<std::string>("group", group_name, "right_arm");
   std::string tip_name;
   pnh.param<std::string>("tip", tip_name, "rh_palm");

   moveit_msgs::RobotTrajectory trajectory;
   trajectory.joint_trajectory.header.stamp = ros::Time::now();
   const auto* jmg{ psm.getRobotModel()->getJointModelGroup(group_name) };
   if(!jmg){
      ROS_FATAL_STREAM("specified group '" << group_name << "' does not exist");
      return 1;
   }
   trajectory.joint_trajectory.joint_names = jmg->getActiveJointModelNames();
   
   return 0;
}
