#include <ros/ros.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/conversions.h>

#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

#include <moveit/move_group_interface/move_group_interface.h>

#include <rviz_visual_tools/remote_control.h>

#include <tf2_ros/transform_listener.h>

#include <bio_ik/goal_types.h>

#include <nav_msgs/Path.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <geometry_msgs/Pose.h>

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
	std::function<void()> publish;
};

void generateTrajectory(trajectory_msgs::JointTrajectory& traj, const GenerateArgs& args){
	// prepare traj
	traj.header.stamp = ros::Time::now();
	const auto* jmg{ args.scene.getRobotModel()->getJointModelGroup(args.group) };
	if(!jmg){
		ROS_FATAL_STREAM("specified group '" << args.group << "' does not exist");
		return;
	}
	traj.joint_names = jmg->getActiveJointModelNames();

	// fill in current state
	traj.points.emplace_back();
	args.scene.getCurrentState().copyJointGroupPositions(
	         jmg,
	         traj.points.back().positions);
	double time_from_start= 0.0;
	traj.points.back().time_from_start= ros::Duration{ time_from_start };
	time_from_start+= 1.0;

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
		         tf2::Vector3{ 0, 0, 1 },
		         tf2::Vector3{ 0, 0, -1 },
		         1.0
		         );
		constraints.add<bio_ik::DirectionGoal>(
		         args.tip,
		         tf2::Vector3{ 1, 0, 0 },
		         tf2::Vector3{ 0, 1, 0 },
		         0.005
		         );

		ik(wp, constraints);

		for(int j= 1; j < 20; ++j){
			moveit::core::RobotState interpolated{ wp };
			previous_wp.interpolate(wp, 0.05*j, interpolated, jmg);

			traj.points.emplace_back();
			interpolated.copyJointGroupPositions(jmg, traj.points.back().positions);
			traj.points.back().time_from_start = ros::Duration{ time_from_start };
			time_from_start+= 1.0;
		}

		traj.points.emplace_back();
		wp.copyJointGroupPositions(jmg, traj.points.back().positions);
		traj.points.back().time_from_start = ros::Duration{ time_from_start };
		time_from_start+= 1.0;

		wp.updateLinkTransforms();
		Eigen::Isometry3d tip_pose_solved { wp.getFrameTransform(args.tip) };
		double translation_max_dimension_distance{ (tip_pose_solved.translation()-expected_tip_position).array().abs().maxCoeff() };
		ROS_INFO_STREAM("distance: " << translation_max_dimension_distance);

		previous_wp = wp;
	}

	{
		trajectory_processing::TimeOptimalTrajectoryGeneration time_parameterization{ 1.0, 0.2 };
		robot_trajectory::RobotTrajectory rtraj{ args.scene.getRobotModel(), args.group };
		rtraj.setRobotTrajectoryMsg(args.scene.getCurrentState(), traj);
		time_parameterization.computeTimeStamps(rtraj);
		moveit_msgs::RobotTrajectory rtraj_msg;
		rtraj.getRobotTrajectoryMsg(rtraj_msg);
		traj = rtraj_msg.joint_trajectory;
	}
}

int main(int argc, char** argv){
   ros::init(argc, argv, "path_to_traj");
   ros::NodeHandle nh, pnh{"~"};

	ros::AsyncSpinner spinner{ 2 };
   spinner.start();

	ros::Publisher pub_traj { nh.advertise<moveit_msgs::DisplayTrajectory>("pluck/trajectory", 5, true) };

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
   std::string tip_name;
	pnh.param<std::string>("tip", tip_name, "rh_fftip");
	if(!scene.getRobotModel()->hasLinkModel(tip_name)){
	  ROS_FATAL_STREAM("LinkModel '" << tip_name << "' does not exist");
	  return 1;
	}

	rviz_visual_tools::RemoteControl remote{ pnh };
	moveit::planning_interface::MoveGroupInterface::Options options{ "right_arm_and_hand" };
	options.robot_model_= psm.getRobotModel();
	options.group_name_ = "right_arm";
	moveit::planning_interface::MoveGroupInterface mgi{ options };
   mgi.setMaxVelocityScalingFactor(1.0);
   mgi.setMaxAccelerationScalingFactor(1.0);

	// move torso to reasonable height
	//{
	//options.group_name_ = "torso";
	//moveit::planning_interface::MoveGroupInterface mgi_torso{ options };
   //mgi_torso.setMaxVelocityScalingFactor(1.0);
   //mgi_torso.setMaxAccelerationScalingFactor(1.0);
	//mgi_torso.setJointValueTarget({0.16825}); // mean height
	//mgi_torso.move();
	//}

	//{
   //options.group_name_ = "right_arm_and_hand";
	//moveit::planning_interface::MoveGroupInterface mgi_right{ options	};
   //mgi_right.setMaxVelocityScalingFactor(1.0);
   //mgi_right.setMaxAccelerationScalingFactor(1.0);

	//mgi_right.setJointValueTarget({-1.048134568510038, 0.33186522463132273, -1.4104755797305828, -1.703852694944122, -0.5134409648277009, -0.315195934553282, 0.07970872374449509, -1.2502830475568716e-06, -0.26175084213630034, 0.8192479550796485, 0.3499906866513662, 8.040438331663623e-05, -2.8072419343516263e-05, 1.5707268944669193, 1.5707889340775059, 0.349921315932693, 3.160236626863484e-05, 1.570694899902314, 1.5707822109633267, 0.34995090350736846, -9.304465432651334e-05, 1.5706199421043197, 1.5707274787708079, 0.3499912777154244, 0.19005917587326865, 0.38993968266188184, 3.916621645912522e-05, 0.5799610561233474, 0.3499207099695695});
	//mgi_right.setJointValueTarget({{"rh_LFJ3", -0.2617261607198903},{"rh_LFJ2", 0.0}}); // broken lfj2
	//mgi_right.move();
	//}

	ros::Subscriber sub{ nh.subscribe<nav_msgs::Path>("pluck/path", 1,
		                                               [&](const auto& path){
		ROS_INFO_STREAM("got path with " << path->poses.size() << " poses");

		ROS_INFO("planning trajectory");
		moveit_msgs::DisplayTrajectory trajectory;
		trajectory.model_id = psm.getRobotModel()->getName();
		trajectory.trajectory.resize(1);
		if(!psm.requestPlanningSceneState()){
			ROS_ERROR_STREAM("failed to get current scene from move_group");
			return;
		}
		moveit::core::robotStateToRobotStateMsg(scene.getCurrentState(), trajectory.trajectory_start, false);

		nav_msgs::Path path_transformed{ *path };
		const auto& planning_frame{ scene.getPlanningFrame() };
		for(auto& pose : path_transformed.poses){
			if(pose.header.frame_id.empty())
				pose.header.frame_id = path_transformed.header.frame_id;
			pose = tf_buffer->transform(pose, planning_frame);
		}

		generateTrajectory(trajectory.trajectory[0].joint_trajectory,
		   {
		   .path = path_transformed,
		   .group = group_name,
		   .tip = tip_name,
			.scene = scene,
			.publish = [&]{ pub_traj.publish(trajectory); }
		   }
		);
		pub_traj.publish(trajectory);
		ROS_INFO_STREAM("publish trajectory with " << trajectory.trajectory[0].joint_trajectory.points.size() << " points");

		remote.waitForNextStep("execute trajectory?");
		ros::Duration(1.0).sleep();

		{
			auto status{ mgi.execute(trajectory.trajectory[0]) };
			ROS_INFO_STREAM("status after execution: " << status);
		}
		})
	};


	ros::waitForShutdown();
   return 0;
}
