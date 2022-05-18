#include <ros/ros.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/conversions.h>

#include <nav_msgs/Path.h>
#include <moveit_msgs/DisplayTrajectory.h>

int main(int argc, char** argv){
   ros::init(argc, argv, "path_to_traj");
   ros::NodeHandle nh, pnh{"~"};

   ros::AsyncSpinner spinner{ 1 };
   spinner.start();

	ros::Publisher pub_traj { nh.advertise<moveit_msgs::DisplayTrajectory>("pluck/trajectory", 1, true) };

   planning_scene_monitor::PlanningSceneMonitor psm{ "robot_description" };
   if(!psm.requestPlanningSceneState()){
      ROS_FATAL_STREAM("failed to get current scene from move_group");
      return 1;
   }
	auto& scene { *psm.getPlanningScene() };
   ROS_INFO("finished loading scene");

   std::string group_name;
   pnh.param<std::string>("group", group_name, "right_arm");
   std::string tip_name;
   pnh.param<std::string>("tip", tip_name, "rh_palm");

	nav_msgs::PathConstPtr path{ ros::topic::waitForMessage<nav_msgs::Path>("pluck/path", nh) };
	ROS_INFO_STREAM("got path with " << path->poses.size() << " poses");

	moveit_msgs::DisplayTrajectory trajectory;
	trajectory.model_id = psm.getRobotModel()->getName();
	trajectory.trajectory.resize(1);
	moveit::core::robotStateToRobotStateMsg(scene.getCurrentState(), trajectory.trajectory_start, false);

	auto& jtrajectory{ trajectory.trajectory[0].joint_trajectory };
	jtrajectory.header.stamp = ros::Time::now();
   const auto* jmg{ psm.getRobotModel()->getJointModelGroup(group_name) };
   if(!jmg){
      ROS_FATAL_STREAM("specified group '" << group_name << "' does not exist");
      return 1;
   }
	jtrajectory.joint_names = jmg->getActiveJointModelNames();

	jtrajectory.points.emplace_back();
	scene.getCurrentState().copyJointGroupPositions(
	         jmg,
	         jtrajectory.points.back().positions);
	jtrajectory.points.back().time_from_start= ros::Duration(0.0);

	for(std::size_t i = 0; i < jtrajectory.points.front().positions.size(); ++i){
		jtrajectory.points.emplace_back(jtrajectory.points.back());
		jtrajectory.points.back().positions[i] = 0.0;
		jtrajectory.points.back().time_from_start= ros::Duration(0.5);
	}

	pub_traj.publish(trajectory);
	ROS_INFO_STREAM("publish trajectory with " << trajectory.trajectory[0].joint_trajectory.points.size() << " points");


	ros::waitForShutdown();
   return 0;
}
