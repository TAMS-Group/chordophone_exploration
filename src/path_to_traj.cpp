#include <ros/ros.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/conversions.h>

#include <nav_msgs/Path.h>
#include <moveit_msgs/DisplayTrajectory.h>

struct GenerateArgs {
	trajectory_msgs::JointTrajectory& trajectory;
	const nav_msgs::Path path;
	const std::string group;
	const std::string tip;
	const planning_scene::PlanningScene& scene;
};

void generateTrajectory(const GenerateArgs& args){
	auto& traj{ args.trajectory };
	traj.header.stamp = ros::Time::now();

	const auto* jmg{ args.scene.getRobotModel()->getJointModelGroup(args.group) };
	if(!jmg){
		ROS_FATAL_STREAM("specified group '" << args.group << "' does not exist");
		return;
	}
	traj.joint_names = jmg->getActiveJointModelNames();

	traj.points.emplace_back();
	args.scene.getCurrentState().copyJointGroupPositions(
	         jmg,
	         traj.points.back().positions);
	traj.points.back().time_from_start= ros::Duration(0.0);

	for(std::size_t i = 0; i < traj.points.front().positions.size(); ++i){
		traj.points.emplace_back(traj.points.back());
		traj.points.back().positions[i] = 0.0;
		traj.points.back().time_from_start= ros::Duration(0.5);
	}
}

int main(int argc, char** argv){
   ros::init(argc, argv, "path_to_traj");
   ros::NodeHandle nh, pnh{"~"};

	ros::AsyncSpinner spinner{ 2 };
   spinner.start();

	ros::Publisher pub_traj { nh.advertise<moveit_msgs::DisplayTrajectory>("pluck/trajectory", 1, true) };

	planning_scene_monitor::PlanningSceneMonitor psm{ "robot_description" };
	auto& scene { *psm.getPlanningScene() };
   ROS_INFO("finished loading scene");

   std::string group_name;
   pnh.param<std::string>("group", group_name, "right_arm");
   std::string tip_name;
   pnh.param<std::string>("tip", tip_name, "rh_palm");

	ros::Subscriber sub{ nh.subscribe<nav_msgs::Path>("pluck/path", 1,
		                                               [&](auto& path){
		ROS_INFO_STREAM("got path with " << path->poses.size() << " poses");

		moveit_msgs::DisplayTrajectory trajectory;
		trajectory.model_id = psm.getRobotModel()->getName();
		trajectory.trajectory.resize(1);
		if(!psm.requestPlanningSceneState()){
			ROS_ERROR_STREAM("failed to get current scene from move_group");
			return;
		}
		moveit::core::robotStateToRobotStateMsg(scene.getCurrentState(), trajectory.trajectory_start, false);
		generateTrajectory({
		   .trajectory = trajectory.trajectory[0].joint_trajectory,
		   .path = *path,
		   .group = group_name,
		   .tip = tip_name,
		   .scene = scene});
		pub_traj.publish(trajectory);
		ROS_INFO_STREAM("publish trajectory with " << trajectory.trajectory[0].joint_trajectory.points.size() << " points");
		})
	};


	ros::waitForShutdown();
   return 0;
}
