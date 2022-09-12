﻿#include <ros/ros.h>

#include <actionlib/server/simple_action_server.h>
#include <tams_pr2_guzheng/ExecutePathAction.h>
using tams_pr2_guzheng::ExecutePathAction;
using tams_pr2_guzheng::ExecutePathGoalConstPtr;
using tams_pr2_guzheng::ExecutePathResult;

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene_monitor/current_state_monitor.h>
#include <moveit/planning_scene_monitor/trajectory_monitor.h>
#include <moveit/robot_state/conversions.h>

#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

#include <moveit/move_group_interface/move_group_interface.h>

#include <rviz_visual_tools/remote_control.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

#include <bio_ik/goal_types.h>

#include <nav_msgs/Path.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <geometry_msgs/Pose.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using std::size_t;

class IkOptions : public bio_ik::BioIKKinematicsQueryOptions {
public:
	using bio_ik::BioIKKinematicsQueryOptions::BioIKKinematicsQueryOptions;

	template<typename Goal, typename... Args>
	void add(Args... args){
		goals.push_back(std::make_unique<Goal>(args...));
	}
};

struct GenerateArgs {
	const nav_msgs::Path& path;
	const std::string group;
	const std::string tip;
	const planning_scene::PlanningScene& scene;
};

robot_trajectory::RobotTrajectory generateTrajectory(const GenerateArgs& args){
	const auto* jmg{ args.scene.getRobotModel()->getJointModelGroup(args.group) };
	// prepare traj
	if(!jmg){
		ROS_FATAL_STREAM("specified group '" << args.group << "' does not exist");
		throw std::runtime_error{"group does not exist"};
	}
	robot_trajectory::RobotTrajectory traj{ args.scene.getRobotModel(), jmg};

	traj.addSuffixWayPoint(args.scene.getCurrentState(), 0.0);

	moveit::core::RobotState wp { args.scene.getCurrentState() }, previous_wp{ wp };

	// fill in useless tip for MoveIt - "replace" is set, so MoveIt's goal is ignored either way
	const std::string arbitrary_tip_link{ [&]{
		std::vector<std::string> eef_tips;
		jmg->getEndEffectorTips(eef_tips);
		return eef_tips.at(0);
	}() };

	// do collision-aware IK from BioIkKinematicsQueryOptions
	auto ik = [&](auto& state, const auto& constraints){
		return state.setFromIK(jmg,
		          geometry_msgs::Pose{},
		          arbitrary_tip_link,
		          0.1,
		          [&args](
		            auto robot_state,
		            auto jmg,
		          const double* positions)
	{
		robot_state->setJointGroupPositions(jmg, positions);
		return !args.scene.isStateColliding(*robot_state, jmg->getName());
	},
		         constraints
		   );
	};

	IkOptions constraints;
	constraints.replace= true;
	constraints.return_approximate_solution = true;

	for(const auto& pose : args.path.poses){
		Eigen::Vector3d expected_tip_position{ pose.pose.position.x, pose.pose.position.y, pose.pose.position.z };

		constraints.goals.clear();
		constraints.add<bio_ik::PositionGoal>(
		         args.tip,
		         tf2::Vector3{expected_tip_position.x(),expected_tip_position.y(),expected_tip_position.z()},
		         1.0
		         );
		constraints.add<bio_ik::DirectionGoal>(
		         args.tip,
		         tf2::Vector3{ 1, 0, 0 },
		         tf2::Vector3{ 0, 0, -1 },
		         1.0
		         );
		constraints.add<bio_ik::DirectionGoal>(
		         args.tip,
		         tf2::Vector3{ 0, 1, 0 },
		         tf2::Vector3{ 0, 1, 0 },
		         0.005
		         );

		ik(wp, constraints);

		for(int j= 1; j < 20; ++j){
			moveit::core::RobotState interpolated{ wp };
			previous_wp.interpolate(wp, 0.05*j, interpolated, jmg);

			traj.addSuffixWayPoint(interpolated, 1.0);
		}

		wp.updateLinkTransforms();
		traj.addSuffixWayPoint(wp, 1.0);

		Eigen::Isometry3d tip_pose_solved { wp.getFrameTransform(args.tip) };
		double translation_max_dimension_distance{ (tip_pose_solved.translation()-expected_tip_position).array().abs().maxCoeff() };
		ROS_INFO_STREAM("distance: " << translation_max_dimension_distance);

		previous_wp = wp;
	}

	{
		constexpr double path_tolerance{ 0.1 };
      constexpr double resample_dt{ 0.05 };
      constexpr double min_angle_change{ 0.001 };
		trajectory_processing::TimeOptimalTrajectoryGeneration time_parameterization{ path_tolerance, resample_dt, min_angle_change };
		if(!time_parameterization.computeTimeStamps(traj)){
			throw std::runtime_error{"could not parameterize path"};
		}
		////slow down to half maximum speed
		//time_parameterization.computeTimeStamps(traj, 0.5, 1.0);
	}
	return traj;
}

struct LinkPathArgs {
	const std::string frame;
	const std::string tip;
	robot_trajectory::RobotTrajectory& trajectory;
	tf2_ros::Buffer& tf;
};
nav_msgs::Path getLinkPath(LinkPathArgs args){
	nav_msgs::Path path;
	path.header.frame_id = args.frame;

	auto& t { args.trajectory };
	double duration_from_start{ 0.0 };
	for(size_t i{ 0 }; i < t.getWayPointCount(); ++i){
		auto& wp { *t.getWayPointPtr(i) };
		wp.updateLinkTransforms();
		Eigen::Isometry3d wp_tip{ wp.getGlobalLinkTransform(args.tip) };
		geometry_msgs::PoseStamped wp_pose;
		wp_pose.header.stamp = ros::Time(duration_from_start);
		duration_from_start+= t.getWayPointDurationFromPrevious(i);
		wp_pose.header.frame_id = wp.getRobotModel()->getRootLinkName();
		wp_pose.pose = tf2::toMsg(wp_tip);
		path.poses.emplace_back( args.tf.transform(wp_pose, args.frame) );
	}
	return path;
}

struct PaintArgs {
	const nav_msgs::Path& requested;
	const nav_msgs::Path& generated;
	const nav_msgs::Path executed;
};
sensor_msgs::Image paintLocalPaths(const PaintArgs& args){
	const int width= 200;
	const int height= 100;
	const double pixel_size= 0.001;

	cv::Mat img{ height, width, CV_8UC3, cv::Scalar(128,128,128) };

	// indicate string position
	cv::circle(img, cv::Point{width/2, height*9/10}, 3, cv::Scalar(0,0,0), 1, cv::LINE_AA);

	auto drawPoses{
		[&](const nav_msgs::Path& path, const cv::Scalar& color){
			bool first{ true };
			cv::Point pt1, pt2;
			for(auto& p : path.poses){
				pt1= pt2;
				pt2= cv::Point(width/2-p.pose.position.y/pixel_size, height-p.pose.position.z/pixel_size-height*1/10);
				if( first ){
					first= false;
					continue;
				}

				ROS_DEBUG_STREAM( "adding line from " << pt1 << " to " << pt2);
				cv::line(img, pt1, pt2, color, 1, cv::LINE_AA);
			}
		}
	};

	drawPoses(args.requested, cv::Scalar{255,0,0});
	drawPoses(args.generated, cv::Scalar{0,255,0});
	drawPoses(args.executed, cv::Scalar{0,0,255});

	std_msgs::Header header;
	cv_bridge::CvImage bridge{ header, sensor_msgs::image_encodings::RGB8, img };
	sensor_msgs::Image img_ros;
	bridge.toImageMsg(img_ros);
	return img_ros;
}

int main(int argc, char** argv){
   ros::init(argc, argv, "pluck_from_path");
   ros::NodeHandle nh, pnh{"~"};

	ros::AsyncSpinner spinner{ 3 };
   spinner.start();

	ros::Publisher pub_traj { nh.advertise<moveit_msgs::DisplayTrajectory>("pluck/trajectory", 1, true) };
	ros::Publisher pub_executed_traj { nh.advertise<moveit_msgs::DisplayTrajectory>("pluck/executed_trajectory", 1, true) };
	ros::Publisher pub_img { nh.advertise<sensor_msgs::Image>("pluck/projected_img", 2, true) };
	ros::Publisher pub_path_commanded { nh.advertise<nav_msgs::Path>("pluck/commanded_path", 1, true) };
	ros::Publisher pub_path_planned { nh.advertise<nav_msgs::Path>("pluck/planned_path", 1, true) };
	ros::Publisher pub_path_executed { nh.advertise<nav_msgs::Path>("pluck/executed_path", 1, true) };

	auto tf_buffer{ std::make_shared<tf2_ros::Buffer>() };
	tf2_ros::TransformListener tf_listener{ *tf_buffer };

	planning_scene_monitor::PlanningSceneMonitor psm{ "robot_description", tf_buffer };

	auto& scene { *psm.getPlanningScene() };
   ROS_INFO("finished loading scene");

   std::string group_name;
   pnh.param<std::string>("group", group_name, "right_arm");
	if(!scene.getRobotModel()->hasJointModelGroup(group_name)){
	  ROS_FATAL_STREAM("JointModelGroup '" << group_name << "' does not exist");
	  return 1;
	}
   std::string finger;
	pnh.param<std::string>("finger", finger, "ff");
   const std::vector<std::string> fingers{ "th", "ff", "mf", "rf", "lf" };
	if(std::find(fingers.begin(), fingers.end(), finger) == fingers.end()){
	  ROS_FATAL_STREAM("finger must be one of th/ff/mf/rf/lf");
	  return 1;
	}

	rviz_visual_tools::RemoteControl remote{ pnh };

	moveit::planning_interface::MoveGroupInterface::Options options{ "right_arm_and_hand" };
	options.robot_model_= psm.getRobotModel();
	options.group_name_ = "right_arm";
	moveit::planning_interface::MoveGroupInterface mgi{ options };
   mgi.setMaxVelocityScalingFactor(1.0);
   mgi.setMaxAccelerationScalingFactor(1.0);

	auto csm { std::make_shared<planning_scene_monitor::CurrentStateMonitor>(scene.getRobotModel(), tf_buffer, nh) };
	csm->enableCopyDynamics(true);
	csm->startStateMonitor();
	planning_scene_monitor::TrajectoryMonitor tm{ csm, 50.0 };

   std::unique_ptr<actionlib::SimpleActionServer<tams_pr2_guzheng::ExecutePathAction>> execute_path_server;

	auto actionCB{ [&, &server= execute_path_server](const tams_pr2_guzheng::ExecutePathGoalConstPtr& goal){
      auto& path{ goal->path };
		ROS_INFO_STREAM("got path with " << path.poses.size() << " poses");

      std::string tip_name{ "rh_" + (goal->finger.empty() ? finger : goal->finger) + "_biotac_link" };
      if(!scene.getRobotModel()->hasLinkModel(tip_name)){
          ROS_ERROR_STREAM("Could not find required tip frame for plucking motion: '" << tip_name << "'.");
          server->setAborted();
          return;
      }

		ROS_INFO("planning trajectory");

		// update scene + start trajectory at current state
		if(!psm.requestPlanningSceneState()){
			ROS_ERROR_STREAM("failed to get current scene from move_group");
			server->setAborted();
			return;
		}

		// transform requested path to planning frame
		nav_msgs::Path path_transformed{ path };
		const auto& planning_frame{ scene.getPlanningFrame() };
		path_transformed.header.frame_id= planning_frame;
		for(auto& pose : path_transformed.poses){
			if(pose.header.frame_id.empty())
		      pose.header.frame_id = path.header.frame_id;
			pose = tf_buffer->transform(pose, planning_frame);
		}
		pub_path_commanded.publish( path_transformed );

		// compute trajectory
		robot_trajectory::RobotTrajectory trajectory{ scene.getRobotModel() };
		try {
			trajectory = generateTrajectory({
			   .path = path_transformed,
			   .group = group_name,
			   .tip = tip_name,
				.scene = scene,
			});
		}
		catch(const std::runtime_error& e){
			ROS_ERROR_STREAM("Failed to generate trajectory: " << e.what());
			server->setAborted();
			return;
		}

		// propagate result
		moveit_msgs::RobotTrajectory trajectory_msg;
		trajectory.getRobotTrajectoryMsg(trajectory_msg);
		{
			moveit_msgs::DisplayTrajectory dtrajectory;
			dtrajectory.model_id = psm.getRobotModel()->getName();
			moveit::core::robotStateToRobotStateMsg(scene.getCurrentState(), dtrajectory.trajectory_start, false);
			dtrajectory.trajectory.reserve(1);
			dtrajectory.trajectory.push_back(trajectory_msg);
			pub_traj.publish(dtrajectory);
			ROS_INFO_STREAM("publish trajectory with " << trajectory.getWayPointCount() << " points");
		}

		// create image to show trajectories
		nav_msgs::Path generated_path { getLinkPath({
			                                        .frame = path.header.frame_id,
			                                        .tip = tip_name,
			                                        .trajectory = trajectory,
			                                        .tf = *tf_buffer
			                                     }) };
		pub_path_planned.publish( generated_path );

		// execute after confirmation from user
		remote.waitForNextStep("execute trajectory?");
		if(!remote.getAutonomous()){
			ros::Duration(1.0).sleep();
		}

		//csm->startStateMonitor();
		tm.clearTrajectory();
		tm.startTrajectoryMonitor();
		auto status{ mgi.execute(trajectory_msg) };
		tm.stopTrajectoryMonitor();
		//csm->stopStateMonitor();
		ROS_INFO_STREAM("status after execution: " << status);
		robot_trajectory::RobotTrajectory executed_trajectory{ tm.getTrajectory() };

		moveit_msgs::RobotTrajectory executed_trajectory_msg;
		if(!executed_trajectory.empty()){
			executed_trajectory.getRobotTrajectoryMsg(executed_trajectory_msg);
			{
				moveit_msgs::DisplayTrajectory dtrajectory;
				dtrajectory.model_id = psm.getRobotModel()->getName();
				moveit::core::robotStateToRobotStateMsg(executed_trajectory.getFirstWayPoint(), dtrajectory.trajectory_start, false);
				dtrajectory.trajectory.reserve(1);
				dtrajectory.trajectory.push_back(executed_trajectory_msg);
				pub_executed_traj.publish(dtrajectory);
			}
		}
		else {
			ROS_ERROR("Recorded trajectory is empty? It shouldn't be. Not publishing anything.");
		}

		nav_msgs::Path executed_path{ getLinkPath({
			                                        .frame = path.header.frame_id,
			                                        .tip = tip_name,
			                                        .trajectory = executed_trajectory,
			                                        .tf = *tf_buffer
			                                     }) };
		pub_img.publish(
		         paintLocalPaths({
		                            .requested = path,
		                            .generated = generated_path,
		                            .executed = executed_path
		                         })
		         );

		pub_path_executed.publish( executed_path );

		ExecutePathResult result;
		result.generated_path = generated_path;
		result.generated_trajectory = trajectory_msg.joint_trajectory;
		result.executed_path = executed_path;
		result.executed_trajectory = executed_trajectory_msg.joint_trajectory;

		server->setSucceeded(result);
	} };
   execute_path_server = std::make_unique<actionlib::SimpleActionServer<tams_pr2_guzheng::ExecutePathAction>>(nh, "pluck/execute_path", actionCB, false);
   execute_path_server->start();

   std::unique_ptr<actionlib::SimpleActionServer<tams_pr2_guzheng::ExecutePathAction>> move_server;

	auto moveActionCB{ [&,&server=move_server](const tams_pr2_guzheng::ExecutePathGoalConstPtr& goal){
      auto& path{ goal->path };
		ROS_INFO_STREAM("got path with " << path.poses.size() << " poses");

      std::string tip_name{ "rh_" + (goal->finger.empty() ? finger : goal->finger) + "_biotac_link" };
      if(!scene.getRobotModel()->hasLinkModel(tip_name)){
          ROS_ERROR_STREAM("Could not find required tip frame for plucking motion: '" << tip_name << "'.");
          server->setAborted();
          return;
      }

		ROS_INFO("planning trajectory");

		// update scene + start trajectory at current state
		if(!psm.requestPlanningSceneState()){
			ROS_ERROR_STREAM("failed to get current scene from move_group");
			server->setAborted();
			return;
		}

		// transform requested path to planning frame
		nav_msgs::Path path_transformed{ path };
		const auto& planning_frame{ scene.getPlanningFrame() };
		for(auto& pose : path_transformed.poses){
			if(pose.header.frame_id.empty())
				pose.header.frame_id = path_transformed.header.frame_id;
			pose = tf_buffer->transform(pose, planning_frame);
		}

		// compute trajectory
		robot_trajectory::RobotTrajectory trajectory{ scene.getRobotModel() };
		try {
			trajectory = generateTrajectory({
			   .path = path_transformed,
			   .group = group_name,
			   .tip = tip_name,
				.scene = scene,
			});
		}
		catch(const std::runtime_error& e){
			ROS_ERROR_STREAM("Failed to generate trajectory: " << e.what());
			server->setAborted();
			return;
		}

		// propagate result
		moveit_msgs::RobotTrajectory trajectory_msg;
		trajectory.getRobotTrajectoryMsg(trajectory_msg);
		{
			moveit_msgs::DisplayTrajectory dtrajectory;
			dtrajectory.model_id = psm.getRobotModel()->getName();
			moveit::core::robotStateToRobotStateMsg(scene.getCurrentState(), dtrajectory.trajectory_start, false);
			dtrajectory.trajectory.reserve(1);
			dtrajectory.trajectory.push_back(trajectory_msg);
			pub_traj.publish(dtrajectory);
			ROS_INFO_STREAM("publish trajectory with " << trajectory.getWayPointCount() << " points");
		}

		auto status{ mgi.execute(trajectory_msg) };
		ROS_INFO_STREAM("status after execution: " << status);

		ExecutePathResult result;

		server->setSucceeded(result);
	} };
   move_server = std::make_unique<actionlib::SimpleActionServer<tams_pr2_guzheng::ExecutePathAction>>(nh, "pluck/goto_start", moveActionCB, false);
   move_server->start();

	ros::waitForShutdown();
	return 0;
}
