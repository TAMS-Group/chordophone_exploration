#include <ros/ros.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include <moveit/robot_state/cartesian_interpolator.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/MoveItErrorCodes.h>

#include <tf2_ros/transform_listener.h>

struct FeelString {

	std::shared_ptr<tf2_ros::Buffer> tf_buffer{ std::make_shared<tf2_ros::Buffer>() };
	tf2_ros::TransformListener tf_listener{ *tf_buffer };
	//TODO: must be a pointer because LockedPlanningSceneRO needs one!
	planning_scene_monitor::PlanningSceneMonitorPtr psm{ std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description", tf_buffer) };

	moveit::planning_interface::MoveGroupInterface mgi;

	FeelString(ros::NodeHandle& pnh) :
	   mgi{ [&,this]{
		   moveit::planning_interface::MoveGroupInterface::Options o{ pnh.param<std::string>("group", "right_arm") };
			o.robot_model_ = psm->getRobotModel();
			return o;
	      }() }
	{
		psm->startSceneMonitor("move_group/monitored_planning_scene");
		psm->requestPlanningSceneState();
		psm->startStateMonitor();
	}

	bool build_trajectory(const planning_scene::PlanningScene& scene, moveit_msgs::RobotTrajectory robot_traj_msg){
		moveit::core::RobotState current_state{ scene.getCurrentState() };
		auto* jmg{ scene.getRobotModel()->getJointModelGroup("right_arm") };
		auto link{ scene.getRobotModel()->getLinkModel("rh_palm") };

		/*
		auto is_valid{ [&](moveit::core::RobotState* state, const moveit::core::JointModelGroup* jmg, const double* positions){
			state->setJointGroupPositions(jmg, positions);
			return !scene.isStateColliding(*state, jmg->getName());
		} };
		*/

		std::vector<moveit::core::RobotStatePtr> traj;
		double fraction_reached{ 0.0 };
		fraction_reached= moveit::core::CartesianInterpolator::computeCartesianPath(
		         &current_state,
		         jmg,
		         traj,
		         link,
		         Eigen::Vector3d::UnitZ(),
		         false,
		         .05,
		         moveit::core::MaxEEFStep{ 0.001 },
		         moveit::core::JumpThreshold{ 100.0 }
		         /*is_valid*/
		         );
		if(fraction_reached < 0.05){
			ROS_ERROR_STREAM("could not plan Cartesian trajectory. Only got " << fraction_reached << ".");
			scene.isStateColliding(jmg->getName(), true);
			ros::shutdown();
			return false;
		}

		robot_trajectory::RobotTrajectory robot_traj{ current_state.getRobotModel(), jmg };
		for(auto& wp : traj)
			robot_traj.addSuffixWayPoint(wp, 0.0);
		trajectory_processing::TimeOptimalTrajectoryGeneration totg{};
		totg.computeTimeStamps(robot_traj);
		robot_traj.getRobotTrajectoryMsg(robot_traj_msg);

		return true;
	}

	void run(){
		moveit_msgs::RobotTrajectory robot_traj_msg;

		planning_scene::PlanningScenePtr scene;
		{
			planning_scene_monitor::LockedPlanningSceneRO locked_scene{ psm };
			scene= planning_scene::PlanningScene::clone(locked_scene);
		}

		if(!build_trajectory(*scene, robot_traj_msg)){
			ROS_ERROR("could not build Cartesian trajectory");
			ros::shutdown();
		}

		ROS_INFO_STREAM("trajectory has "  << robot_traj_msg.joint_trajectory.points.size() << " waypoints.");

		// TODO: implement
		//	moveit_msgs::DisplayRobotTrajectory display;

		auto result { mgi.execute(robot_traj_msg) };
		if(result != moveit_msgs::MoveItErrorCodes::SUCCESS){
			ROS_INFO("found string and stopped near it");
		}
		else {
			ROS_INFO("apparently did not touch string this try");
		}
	}
};

int main(int argc, char** argv){
	ros::init(argc, argv, "feel_string");

	ros::NodeHandle nh{"~"};
	ros::AsyncSpinner spinner{ 2 };

	spinner.start();

	FeelString fs{nh};
	fs.run();

	//ros::waitForShutdown();

	return 0;
}
