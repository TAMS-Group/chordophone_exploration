#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit_msgs/RobotTrajectory.h>

moveit::planning_interface::MoveGroupInterfacePtr mgi;

void event_cb(const ros::TimerEvent& event){
  ROS_INFO("Aborting");
  mgi->stop();
}

int main(int argc, char** argv){
  ros::init(argc, argv, "detect_strings");
  ros::AsyncSpinner spinner{ 2 };
  spinner.start();
  ros::NodeHandle nh{"~"};

  mgi = std::make_shared<moveit::planning_interface::MoveGroupInterface>("right_arm_and_hand");

  auto pose { mgi->getCurrentPose("rh_palm") };
  ROS_INFO_STREAM("got palm pose in frame '" << pose.header.frame_id << "':\n" << pose.pose);

  const double distance{ nh.param("distance", 0.3) };
  pose.pose.position.x+= distance;

  moveit_msgs::RobotTrajectory trajectory;
  double fraction{ mgi->computeCartesianPath({ pose.pose }, .02, 1.5, trajectory) };

  ROS_INFO_STREAM("planned Cartesian path for " << distance * fraction << "m forward with " << trajectory.joint_trajectory.points.size() << " waypoints");
  ROS_INFO_STREAM("scheduling abort");
  ros::Timer timer{ nh.createTimer(ros::Duration(0.8), event_cb, true) };
  ROS_INFO_STREAM("executing");
  auto state{ mgi->execute(trajectory) };

  ROS_INFO_STREAM("execution state after return: " << state);

  mgi.reset();
  return 0;
}
