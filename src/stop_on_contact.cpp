#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>

#include <deque>

const ros::Duration FILTER_WIDTH{ 5.0 };
constexpr double STOP_THRESHOLD{ 10.0 }; // of filtered signal

struct DetectContact {
  ros::Publisher pub;
  ros::Publisher pub_cmd;
  ros::Subscriber sub;

  DetectContact(ros::NodeHandle& nh){
    pub = nh.advertise<std_msgs::Float32>("detect_contact", 1);
    pub_cmd = nh.advertise<std_msgs::String>("trajectory_execution_event", 1);
    sub = nh.subscribe("hand/rh/tactile", 1, &DetectContact::getReadings, this);
  }

  struct Point {
	 ros::Time stamp;
	 short data;
  };
  std::deque<Point> last_values;

  void getReadings(const sr_robot_msgs::BiotacAll& tacs){
	 last_values.push_back({tacs.header.stamp, tacs.tactiles[0].pdc});

	 if(last_values.front().stamp + FILTER_WIDTH < last_values.back().stamp){
      std_msgs::Float32 data;
		data.data= last_values.back().data - last_values.front().data;
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
