#ifndef PKUN_TELEOP__JOY_TELEOP_NODE_HPP_
#define PKUN_TELEOP__JOY_TELEOP_NODE_HPP_

#include <map>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "pkun_msgs/msg/body_pose.hpp"
#include "pkun_msgs/msg/gesture.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"

namespace pkun_teleop
{

/// Bluetooth gamepad -> robot intent.
///
/// The Bluetooth link itself is not this node's job. Pair the pad once with
/// bluetoothctl, let the kernel expose it as /dev/input/jsN, and run the stock
/// `joy_linux` node to turn that into sensor_msgs/Joy. This node subscribes to
/// Joy, so it works identically with a USB pad, a BT pad or a joy bag replay.
///
/// Safety model: the left trigger is a deadman. Walking commands are only
/// emitted while it is held, and a zero Twist is sent the moment it is
/// released. Gesture buttons work without it, since gestures do not translate
/// the robot.
class JoyTeleopNode : public rclcpp::Node
{
public:
  explicit JoyTeleopNode(const rclcpp::NodeOptions & options);

private:
  void on_joy(const sensor_msgs::msg::Joy::SharedPtr msg);
  void on_timeout_check();

  double axis(const sensor_msgs::msg::Joy & joy, int index, double scale) const;
  bool pressed(const sensor_msgs::msg::Joy & joy, int index) const;
  bool rising_edge(const sensor_msgs::msg::Joy & joy, int index);

  // axis / button mapping (defaults suit an Xbox-layout pad)
  int axis_linear_x_{1};
  int axis_linear_y_{0};
  int axis_angular_z_{3};
  int axis_body_pitch_{4};
  int deadman_button_{6};
  int posture_mode_button_{4};

  double scale_linear_x_{60.0};    // mm/s
  double scale_linear_y_{40.0};
  double scale_angular_z_{0.8};    // rad/s
  double scale_body_pitch_{0.26};  // rad
  double deadzone_{0.12};

  double stand_height_{110.0};
  double joy_timeout_{1.0};

  std::map<int, std::string> gesture_buttons_;
  std::vector<int> previous_buttons_;
  bool deadman_held_{false};
  bool sent_stop_{true};
  rclcpp::Time last_joy_stamp_;

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<pkun_msgs::msg::Gesture>::SharedPtr gesture_pub_;
  rclcpp::Publisher<pkun_msgs::msg::BodyPose>::SharedPtr body_pose_pub_;
  rclcpp::TimerBase::SharedPtr watchdog_;
};

}  // namespace pkun_teleop

#endif  // PKUN_TELEOP__JOY_TELEOP_NODE_HPP_
