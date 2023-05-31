#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <dynamic_reconfigure/server.h>

#include <tams_pr2_guzheng/ThresholdConfig.h>
#include <tams_pr2_guzheng/TactilePluck.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_variance.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>

namespace acc = boost::accumulators;

struct DetectContact {
  using Accumulator = acc::accumulator_set<
    double, 
    acc::stats<
      // acc::tag::rolling_window // TODO: broken in debian testing
      acc::tag::rolling_window_plus1
      // acc::tag::rolling_mean,
      // acc::tag::rolling_variance
    > >;

  Accumulator aggregator { acc::tag::rolling_window::window_size = 3 };
  size_t last_event { 4 }; // hack to ensure full buffer before starting detection

  // configuration
  size_t wait { 10 }; // 100ms in 0.01ms samples
  double threshold { 10.0 };

  /* 
    @param value: the new value to be added to the aggregator
    @param process_signal: the process value thresholded to detect a pluck
    @return a strength value > 0.0 if a pluck was detected
  */
  double operator()(double value, double& process_signal){
    aggregator(value);

    // if (!acc::impl::is_rolling_window_plus1_full(aggregator))
    //   return 0.0;

    if(last_event){
          --last_event;
          return 0.0;
    }

    // if aggregators are not full yet, we don't have a valid value
    // TODO: does not compile
    // if(!acc::impl::is_rolling_window_plus1_full(aggregators[0]))
    //   return false;

    auto window{ acc::rolling_window_plus1(aggregator) };
    if( boost::empty(window) )
      return 0.0;

    process_signal = window.front()-window.back();
    if(process_signal > threshold){
      last_event = wait;
      return process_signal;
    }

    // alternative to detect significant positive outliers in process_signal
    // double sample{ static_cast<double>(tacs.tactiles[i].pdc) };
    // double mean{ acc::rolling_mean(aggregators[i]) };
    // double stddev{ std::sqrt(acc::rolling_variance(aggregators[i])) };
    // signals.data.push_back(mean);
    // if (std::abs(mean - sample) > stddev*3.5){
    //   // found outlier, track a pluck

    return 0.0;
  }
};


struct DetectContactHand {
  static constexpr std::array<char const*, 5> fingers{ "ff", "mf", "rf", "lf", "th"};

  std::array<DetectContact, fingers.size()> detectors;

  ros::NodeHandle nh;
  ros::NodeHandle pnh;

  ros::Publisher pub_signals;
  ros::Publisher pub_detect;
  ros::Publisher pub_plucks;
  ros::Subscriber sub;

  dynamic_reconfigure::Server<tams_pr2_guzheng::ThresholdConfig> config_server;
  tams_pr2_guzheng::ThresholdConfig config;

  ros::Time last_added_sample;

  DetectContactHand() : nh(), pnh("~") {
    for (size_t i = 0; i < fingers.size(); ++i)
      detectors[i] = DetectContact{};

    pub_signals = pnh.advertise<std_msgs::Float64MultiArray>("signal", 1);
    pub_detect = pnh.advertise<std_msgs::Float64MultiArray>("detection", 1);
    pub_plucks = nh.advertise<tams_pr2_guzheng::TactilePluck>("plucks", 10);

    config_server.setCallback(
      [this](tams_pr2_guzheng::ThresholdConfig c, uint32_t lvl){
        this->config_cb(c,lvl);
      }
    );

    sub = nh.subscribe("hand/rh/tactile", 1, &DetectContactHand::getReadings, this);
  }

  void config_cb(tams_pr2_guzheng::ThresholdConfig& c, uint32_t level){
    if(c.wait != config.wait){
      for(auto& d : detectors)
        d.wait = c.wait;
    }

    if(c.threshold != config.threshold){
      for(auto& d : detectors)
        d.threshold = c.threshold;
    }

    config= c;
  }

  void getReadings(const sr_robot_msgs::BiotacAll& tacs){
      // with 1khz update rate (our PR2), we get 10 times the same pdc value in a row, so we skip the redundant ones
    if( tacs.header.stamp - last_added_sample < ros::Duration(0.01))
      return;

    assert(tacs.tactiles.size() == fingers.size());

    std_msgs::Float64MultiArray signals, detections;
    signals.data.resize(fingers.size());
    detections.data.resize(fingers.size());
    std::array<double, fingers.size()> pluck_strength;

    for (size_t i = 0; i < fingers.size(); ++i){
      pluck_strength[i] = detectors[i](tacs.tactiles[i].pdc, signals.data[i]);
      detections.data[i] = pluck_strength[i] > 0.0 ? 1.0 : 0.0;
      if (pluck_strength[i] > 0.0){
        tams_pr2_guzheng::TactilePluck pluck;
        pluck.finger = fingers[i];
        pluck.header.stamp = tacs.header.stamp;
        pluck.strength = pluck_strength[i];
        pub_plucks.publish(pluck);
      }
    }

    pub_signals.publish(signals);
    pub_detect.publish(detections);
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "detect_pluck");

  DetectContactHand dc;

  ros::spin();

  return 0;
}
