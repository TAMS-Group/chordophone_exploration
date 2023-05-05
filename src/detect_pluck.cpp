#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <dynamic_reconfigure/server.h>
#include <tams_pr2_guzheng/ThresholdConfig.h>

#include <deque>

const ros::Duration FILTER_WIDTH{ 0.02 };

struct DetectContact {
  ros::Publisher pub;
  ros::Publisher pub_detect;
  ros::Publisher pub_event;
  ros::Subscriber sub;

  ros::Time last_event;

  dynamic_reconfigure::Server<tams_pr2_guzheng::ThresholdConfig> config_server;

  int finger_idx;
  tams_pr2_guzheng::ThresholdConfig config;

  void config_cb(tams_pr2_guzheng::ThresholdConfig& c, uint32_t level){
    config= c;
  }

  DetectContact(ros::NodeHandle& nh, ros::NodeHandle& pnh){
    pnh.param<int>("finger_idx", finger_idx, 0);
    assert(finger_idx >= 0 && finger_idx < 5);
    pub = pnh.advertise<std_msgs::Float32>("signal", 1);
    pub_detect = pnh.advertise<std_msgs::Float32>("detection", 1);
    pub_event = nh.advertise<visualization_msgs::MarkerArray>("plucks", 10);

    config_server.setCallback([this](tams_pr2_guzheng::ThresholdConfig c, uint32_t lvl){ this->config_cb(c,lvl); });

    latest_values.push_back({ros::Time(0.0), 0});
    sub = nh.subscribe("hand/rh/tactile", 1, &DetectContact::getReadings, this);
  }

  struct Point {
     ros::Time stamp;
     short data;
  };
  std::deque<Point> latest_values;

  void getReadings(const sr_robot_msgs::BiotacAll& tacs){
    double value;

    // with 1khz update rate (our PR2), we get 10 times the same pdc value in a row, so we skip the redundant ones
    if( tacs.header.stamp - latest_values.back().stamp < ros::Duration(0.01))
      return;

    latest_values.push_back({tacs.header.stamp, tacs.tactiles.at(finger_idx).pdc});

    if(latest_values.front().stamp + FILTER_WIDTH > tacs.header.stamp)
      return;

    value= std::abs(latest_values.back().data - latest_values.front().data);
    std_msgs::Float32 data;
    data.data= value;
    pub.publish(data);
    while(latest_values.front().stamp + FILTER_WIDTH < tacs.header.stamp)
      latest_values.pop_front();

    if(value > config.threshold && (last_event+ros::Duration(config.wait)) < tacs.header.stamp){
      last_event = tacs.header.stamp;
      ROS_INFO_STREAM("detected event (sleeping for " << config.wait << "s)");
      visualization_msgs::MarkerArray ma;
      ma.markers.emplace_back([&]{
        visualization_msgs::Marker m;
        m.ns= "tactile_pluck";
        m.header.stamp= tacs.header.stamp;
        m.action= visualization_msgs::Marker::ADD;
        m.type= visualization_msgs::Marker::CUBE;
        m.scale.x= 0.005;
        m.scale.y= 0.005;
        m.scale.z= 0.005;
        m.color.r= 0.4;
        m.color.g= 0.4;
        m.color.b= 0.4;
        m.color.a= 1.0;

        return m;
      }());
      pub_event.publish(ma);

      std_msgs::Float32 data;
		data.data= 1.0;
      pub_detect.publish(data);
    }
    else {
      std_msgs::Float32 data;
		data.data= 0.0;
      pub_detect.publish(data);
    }
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "detect_pluck");

  ros::NodeHandle nh{}, pnh{"~"};

  DetectContact dc{ nh, pnh };

  ros::spin();

  return 0;
}
