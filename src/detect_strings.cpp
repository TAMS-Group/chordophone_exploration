#include <ros/ros.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "detect_strings");

  ros::spin();
  return 0;
}
