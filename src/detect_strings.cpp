#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit_msgs/RobotTrajectory.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "detect_strings");
  ros::AsyncSpinner spinner{ 1 };
  spinner.start();
  ros::NodeHandle nh{"~"};

  moveit::planning_interface::MoveGroupInterface mgi{ "right_arm_and_hand" };

  auto pose { mgi.getCurrentPose("rh_palm") };
  ROS_INFO_STREAM("got palm pose in frame '" << pose.header.frame_id << "':\n" << pose.pose);

  const double distance{ nh.param("distance", 0.1) };
  pose.pose.position.x + distance;

  moveit_msgs::RobotTrajectory trajectory;
  double fraction{ mgi.computeCartesianPath({ pose.pose }, .02, 1.5, trajectory) };

  ROS_INFO_STREAM("planned Cartesian path for " << distance * fraction << "m forward with " << trajectory.joint_trajectory.points.size() << " waypoints");
  mgi.execute(trajectory);

  return 0;
}
