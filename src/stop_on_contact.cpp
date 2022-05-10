#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>

#include <deque>

constexpr size_t FILTER_THRESHOLD{ 10 };
constexpr double STOP_THRESHOLD{ 10.0 };

struct DetectContact {
  ros::Publisher pub;
  ros::Publisher pub_cmd;
  ros::Subscriber sub;

  DetectContact(ros::NodeHandle& nh){
    pub = nh.advertise<std_msgs::Float32>("detect_contact", 1);
    pub_cmd = nh.advertise<std_msgs::String>("trajectory_execution_event", 1);
    sub = nh.subscribe("hand/rh/tactile", 1, &DetectContact::getReadings, this);
  }

  std::deque<double> last_values;
  void getReadings(const sr_robot_msgs::BiotacAll& tacs){
    last_values.push_back(tacs.tactiles[0].pdc);
    if(last_values.size() > FILTER_THRESHOLD){
      std_msgs::Float32 data;
      data.data= last_values.back() - last_values.front();
      pub.publish(data);
      last_values.pop_front();

      if(data.data > STOP_THRESHOLD){
        std_msgs::String msg;
        msg.data = "stop";
        pub_cmd.publish(msg);
      }
    }
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "stop_on_contact");

  ros::NodeHandle nh{};

  DetectContact dc{ nh };

  ros::spin();

  return 0;
}
