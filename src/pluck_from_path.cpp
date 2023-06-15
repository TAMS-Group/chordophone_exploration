#include <ros/ros.h>

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
#include <std_msgs/String.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TAU (2.0*M_PI)

using std::size_t;

class IkOptions : public bio_ik::BioIKKinematicsQueryOptions {
public:
	using bio_ik::BioIKKinematicsQueryOptions::BioIKKinematicsQueryOptions;

	template<typename Goal, typename... Args>
  IkOptions& add(Args... args){
		goals.push_back(std::make_unique<Goal>(args...));
    return *this;
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
		          0.03,
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

  double is_thumb{ args.tip == "rh_th_plectrum" ? -1.0 : 1.0 };

	for(const auto& pose : args.path.poses){
		Eigen::Vector3d expected_tip_position{ pose.pose.position.x, pose.pose.position.y, pose.pose.position.z };

		constraints.goals.clear();
    constraints
        .add<bio_ik::PositionGoal>(
		         args.tip,
		         tf2::Vector3{expected_tip_position.x(),expected_tip_position.y(),expected_tip_position.z()},
		         1.0
             )
        // plectrum should point down
        .add<bio_ik::DirectionGoal>(
		         args.tip,
		         tf2::Vector3{ 1, 0, 0 },
		         tf2::Vector3{ sin(TAU/2), 0, cos(TAU/2) },
		         1.0
             )
        // plectrum should hit string with the flat side
        .add<bio_ik::DirectionGoal>(
		         args.tip,
		         tf2::Vector3{ 0, is_thumb, 0 },
		         tf2::Vector3{ 0, 1, 0 },
		         0.005
             )
        // regularize to keep shoulder at central height
        .add<bio_ik::JointVariableGoal>(
             "r_shoulder_lift_joint",
             0.52,
             0.005
          );

		ik(wp, constraints);

		// TODO: interpolate if and only if joint space distance is high
		//for(int j= 1; j < 20; ++j){
		//	moveit::core::RobotState interpolated{ wp };
		//	previous_wp.interpolate(wp, 0.05*j, interpolated, jmg);

		//	traj.addSuffixWayPoint(interpolated, 1.0);
		//}

		wp.updateLinkTransforms();
		traj.addSuffixWayPoint(wp, 1.0);

		// Eigen::Isometry3d tip_pose_solved { wp.getFrameTransform(args.tip) };
		// double translation_max_dimension_distance{ (tip_pose_solved.translation()-expected_tip_position).array().abs().maxCoeff() };
		// ROS_INFO_STREAM("distance: " << translation_max_dimension_distance);

		previous_wp = wp;
	}

	{
		constexpr double path_tolerance{ 0.1 };
      constexpr double resample_dt{ 0.05 };
      constexpr double min_angle_change{ 0.005 };
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
		duration_from_start+= std::max(0.0, t.getWayPointDurationFromPrevious(i));
		wp_pose.header.frame_id = wp.getRobotModel()->getRootLinkName();
		wp_pose.pose = tf2::toMsg(wp_tip);

		// cache stamp because these are from trajectory start (not determined yet)
		// instead of 1970 and that breaks TF.
		auto wp_pose_stamp{ wp_pose.header.stamp };
		wp_pose.header.stamp= ros::Time();
		wp_pose= args.tf.transform(wp_pose, args.frame);
		wp_pose.header.stamp= wp_pose_stamp;
		path.poses.emplace_back( wp_pose );
	}
	return path;
}

struct PaintArgs {
	const nav_msgs::Path& requested;
	const nav_msgs::Path& generated;
	const nav_msgs::Path& executed;
   const std::string& label;
};
sensor_msgs::Image paintLocalPaths(const PaintArgs& args){
	const int width= 400;
	const int height= 200;
	const double pixel_size= 0.0003;

	cv::Mat img{ height, width, CV_8UC3, cv::Scalar(128,128,128) };

	// indicate string position
	cv::circle(img, cv::Point{width/2, height*3/4}, 3, cv::Scalar(0,0,0), 1, cv::LINE_AA);

   cv::putText(img, args.label, cv::Point(0, height-10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,0), 1, cv::LINE_AA);

	auto drawPoses{
		[&](const nav_msgs::Path& path, const cv::Scalar& color){
			bool first{ true };
			cv::Point pt1, pt2;
			for(auto& p : path.poses){
				pt1= pt2;
				pt2= cv::Point(width/2-p.pose.position.y/pixel_size, height-p.pose.position.z/pixel_size-height*1/4);
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


class Server {
  ros::NodeHandle nh_{};
  ros::NodeHandle pnh_{ "~" };
  actionlib::SimpleActionServer<tams_pr2_guzheng::ExecutePathAction> pluck_{ nh_, "pluck/pluck", [this](auto& goal){ pluck(goal); }, false };
  actionlib::SimpleActionServer<tams_pr2_guzheng::ExecutePathAction> execute_path_{ nh_, "pluck/execute_path", [this](auto& goal){ execute_path(goal); }, false};

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_{ std::make_shared<tf2_ros::Buffer>(ros::Duration(30.0)) };
  tf2_ros::TransformListener tf_listener_{ *tf_buffer_ };
  planning_scene_monitor::PlanningSceneMonitor psm_{ "robot_description", tf_buffer_ };
  planning_scene::PlanningScene& scene_{ *psm_.getPlanningScene() };
  const moveit::core::RobotModel& model_{ *scene_.getRobotModel() };
  std::shared_ptr<planning_scene_monitor::CurrentStateMonitor> csm_ {
    [this]{
      auto m= std::make_shared<planning_scene_monitor::CurrentStateMonitor>(scene_.getRobotModel(), tf_buffer_, nh_);
      m->enableCopyDynamics(true);
      m->startStateMonitor();
      m->waitForCompleteState(10.0);
      return m;
    }()
  };
  planning_scene_monitor::TrajectoryMonitor tm{ csm_, 100.0 };

  ros::Publisher pub_finger_{ nh_.advertise<std_msgs::String>("pluck/active_finger", 1) };

  ros::Publisher pub_traj_{ nh_.advertise<moveit_msgs::DisplayTrajectory>("pluck/trajectory", 1, true) };
  ros::Publisher pub_executed_traj_{ nh_.advertise<moveit_msgs::DisplayTrajectory>("pluck/executed_trajectory", 1, true) };
  ros::Publisher pub_img_{ nh_.advertise<sensor_msgs::Image>("pluck/projected_img", 2, true) };

  ros::Publisher pub_path_commanded_{ nh_.advertise<nav_msgs::Path>("pluck/commanded_path", 1, true) };
  ros::Publisher pub_path_planned_{ nh_.advertise<nav_msgs::Path>("pluck/planned_path", 1, true) };
  ros::Publisher pub_path_executed_{ nh_.advertise<nav_msgs::Path>("pluck/executed_path", 1, true) };

  rviz_visual_tools::RemoteControl remote{ nh_ };

  std::string finger_{
    [this]{
      std::string f;
      pnh_.param<std::string>("finger", f, "ff");
      const std::vector<std::string> fingers{ "th", "ff", "mf", "rf", "lf" };
      if(std::find(fingers.begin(), fingers.end(), f) == fingers.end()){
        ROS_FATAL_STREAM("finger must be one of th/ff/mf/rf/lf, but is '" << finger_ << "'");
        throw std::runtime_error{"invalid finger specified"};
      }
      return f;
    }()
  };

  std::string group_name_{
    [this]{
      std::string g;
      pnh_.param<std::string>("group", g, "right_arm");
      if(!model_.hasJointModelGroup(g)){
        ROS_FATAL_STREAM("JointModelGroup '" << group_name_ << "' does not exist");
        throw std::runtime_error{"invalid group specified"};;
      }
      return g;
    }()
  };

  moveit::planning_interface::MoveGroupInterface mgi_{
    [this]{
      moveit::planning_interface::MoveGroupInterface::Options o{ "right_arm_and_hand" };
      o.robot_model_= scene_.getRobotModel();
      o.group_name_ = group_name_;

      moveit::planning_interface::MoveGroupInterface mgi{ o };
      mgi.setMaxVelocityScalingFactor(1.0);
      mgi.setMaxAccelerationScalingFactor(1.0);
      return mgi;
    }()
  };

public:
  Server()
  {
    execute_path_.start();
    pluck_.start();
    ROS_INFO_STREAM("initialized pluck_from_path");
  }

  bool update_scene(){
    // update scene + start trajectory at current state
    if(!psm_.requestPlanningSceneState()){
      ROS_ERROR_STREAM("failed to get current scene from move_group");
      return false;
    }
    csm_->setToCurrentState(scene_.getCurrentStateNonConst());
    return true;
  }

  void pluck(const tams_pr2_guzheng::ExecutePathGoalConstPtr& goal){
    auto& path{ goal->path };
    ROS_INFO_STREAM("got path with " << path.poses.size() << " poses");

    std::string finger_name{ goal->finger.empty() ? finger_ : goal->finger };
    std::string tip_name{ "rh_" + finger_name + "_plectrum" };
    if(!model_.hasLinkModel(tip_name)){
      ROS_ERROR_STREAM("Could not find required tip frame for plucking motion: '" << tip_name << "'.");
      pluck_.setAborted();
      return;
    }

    if(!update_scene()){
      pluck_.setAborted();
      return;
    }

    {
      std_msgs::String finger_name_msg;
      finger_name_msg.data = finger_name;
      pub_finger_.publish(finger_name_msg);
    }

    // transform requested path to planning frame
    nav_msgs::Path path_transformed{ path };
    const auto& planning_frame{ scene_.getPlanningFrame() };
    path_transformed.header.frame_id= planning_frame;
    for(auto& pose : path_transformed.poses){
      if(pose.header.frame_id.empty())
        pose.header.frame_id = path.header.frame_id;
      try {
        pose = tf_buffer_->transform(pose, planning_frame);
      } catch(const tf2::LookupException& e){
        ROS_ERROR_STREAM(e.what());
        pluck_.setAborted();
        return;
      }
    }
    pub_path_commanded_.publish( path ); // publish only now to make sure transform succeeded

    // compute joint trajectory
    robot_trajectory::RobotTrajectory trajectory{ scene_.getRobotModel() };
    try {
      trajectory = generateTrajectory({
                                        .path = path_transformed,
                                        .group = group_name_,
                                        .tip = tip_name,
                                        .scene = scene_,
                                      });
    }
    catch(const std::runtime_error& e){
      ROS_ERROR_STREAM("Failed to generate trajectory: " << e.what());
      pluck_.setAborted();
      return;
    }
    ROS_INFO_STREAM("generated trajectory with " << trajectory.getWayPointCount() << " points");

    // publish visualizations of generated trajectory
    moveit_msgs::RobotTrajectory trajectory_msg;
    trajectory.getRobotTrajectoryMsg(trajectory_msg);
    {
      moveit_msgs::DisplayTrajectory dtrajectory;
      dtrajectory.model_id = model_.getName();
      moveit::core::robotStateToRobotStateMsg(scene_.getCurrentState(), dtrajectory.trajectory_start, false);
      dtrajectory.trajectory.reserve(1);
      dtrajectory.trajectory.push_back(trajectory_msg);
      pub_traj_.publish(dtrajectory);
    }

    // publish planned Cartesian path
    nav_msgs::Path generated_path { getLinkPath({
                                                  .frame = path.header.frame_id,
                                                  .tip = tip_name,
                                                  .trajectory = trajectory,
                                                  .tf = *tf_buffer_
                                                }) };
    pub_path_planned_.publish( generated_path );

    // execute after confirmation from user
    remote.waitForNextStep("execute trajectory?");
    if(!remote.getAutonomous()){
      ros::Duration(1.0).sleep(); // sleep a moment to give the user time to switch their attention
    }
    //csm->startStateMonitor();
    tm.clearTrajectory();
    tm.startTrajectoryMonitor();
    auto status{ mgi_.execute(trajectory_msg) };
    tm.stopTrajectoryMonitor();
    //csm->stopStateMonitor();
    ROS_INFO_STREAM("status after execution: " << status);

    // publish executed joint trajectory
    robot_trajectory::RobotTrajectory executed_trajectory{ tm.getTrajectory() };
    moveit_msgs::RobotTrajectory executed_trajectory_msg;
    if(!executed_trajectory.empty()){
      executed_trajectory.getRobotTrajectoryMsg(executed_trajectory_msg);
      {
        moveit_msgs::DisplayTrajectory dtrajectory;
        dtrajectory.model_id = model_.getName();
        moveit::core::robotStateToRobotStateMsg(executed_trajectory.getFirstWayPoint(), dtrajectory.trajectory_start, false);
        dtrajectory.trajectory.reserve(1);
        dtrajectory.trajectory.push_back(executed_trajectory_msg);
        pub_executed_traj_.publish(dtrajectory);
      }
    }
    else {
      ROS_ERROR("Recorded trajectory is empty? It shouldn't be. Not publishing anything.");
    }

    // publish executed Cartesian path
    nav_msgs::Path executed_path{ getLinkPath({
                                                .frame = path.header.frame_id,
                                                .tip = tip_name,
                                                .trajectory = executed_trajectory,
                                                .tf = *tf_buffer_
                                              }) };
    pub_path_executed_.publish( executed_path );

    // publish image of all paths in 2d projection in string frame
    std::string string_name{ path.header.frame_id };
    string_name = string_name.substr(string_name.find("/")+1);
    string_name = string_name.substr(0, string_name.find("/"));
    pub_img_.publish(
          paintLocalPaths({
                            .requested = path,
                            .generated = generated_path,
                            .executed = executed_path,
                            .label = string_name
                          })
          );

    // finish up action
    ExecutePathResult result;
    result.generated_path = generated_path;
    result.generated_trajectory = trajectory_msg.joint_trajectory;
    result.executed_path = executed_path;
    result.executed_trajectory = executed_trajectory_msg.joint_trajectory;
    pluck_.setSucceeded(result);
  }

  void execute_path(const tams_pr2_guzheng::ExecutePathGoalConstPtr& goal){
    auto& path{ goal->path };
    ROS_INFO_STREAM("got path with " << path.poses.size() << " poses");

    std::string finger_name{ goal->finger.empty() ? finger_ : goal->finger };
    std::string tip_name{ "rh_" + finger_name + "_plectrum" };
    if(!model_.hasLinkModel(tip_name)){
      ROS_ERROR_STREAM("Could not find required tip frame for plucking motion: '" << tip_name << "'.");
      execute_path_.setAborted();
      return;
    }

    if(!update_scene()){
      execute_path_.setAborted();
      return;
    }

    {
      std_msgs::String finger_name_msg;
      finger_name_msg.data = finger_name;
      pub_finger_.publish(finger_name_msg);
    }

    // transform requested path to planning frame
    nav_msgs::Path path_transformed{ path };
    const auto& planning_frame{ scene_.getPlanningFrame() };
    for(auto& pose : path_transformed.poses){
      if(pose.header.frame_id.empty())
        pose.header.frame_id = path_transformed.header.frame_id;
      try {
        pose = tf_buffer_->transform(pose, planning_frame);
      } catch(const tf2::LookupException& e){
        ROS_ERROR_STREAM(e.what());
        execute_path_.setAborted();
        return;
      }
    }

    // compute trajectory
    robot_trajectory::RobotTrajectory trajectory{ scene_.getRobotModel() };
    try {
      trajectory = generateTrajectory({
                                        .path = path_transformed,
                                        .group = group_name_,
                                        .tip = tip_name,
                                        .scene = scene_,
                                      });
    }
    catch(const std::runtime_error& e){
      ROS_ERROR_STREAM("Failed to generate trajectory: " << e.what());
      execute_path_.setAborted();
      return;
    }
    ROS_INFO_STREAM("generated trajectory with " << trajectory.getWayPointCount() << " points");

    // propagate result
    moveit_msgs::RobotTrajectory trajectory_msg;
    trajectory.getRobotTrajectoryMsg(trajectory_msg);
    {
      moveit_msgs::DisplayTrajectory dtrajectory;
      dtrajectory.model_id = model_.getName();
      moveit::core::robotStateToRobotStateMsg(scene_.getCurrentState(), dtrajectory.trajectory_start, false);
      dtrajectory.trajectory.reserve(1);
      dtrajectory.trajectory.push_back(trajectory_msg);
      pub_traj_.publish(dtrajectory);
    }

    remote.waitForNextStep("execute trajectory?");
    if(!remote.getAutonomous()){
      ros::Duration(1.0).sleep();
    }

    auto status{ mgi_.execute(trajectory_msg) };
    ROS_INFO_STREAM("status after execution: " << status);

    ExecutePathResult result;
    result.generated_trajectory = trajectory_msg.joint_trajectory;
    execute_path_.setSucceeded(result);
  }
};

int main(int argc, char** argv){
   ros::init(argc, argv, "pluck_from_path");
   ros::NodeHandle nh, pnh{"~"};

	ros::AsyncSpinner spinner{ 3 };
  spinner.start();

  Server server;

  ros::waitForShutdown();
	return 0;
}
