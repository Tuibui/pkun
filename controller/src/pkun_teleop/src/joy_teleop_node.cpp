#include "pkun_teleop/joy_teleop_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace pkun_teleop
{

JoyTeleopNode::JoyTeleopNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("joy_teleop", options), last_joy_stamp_(0, 0, RCL_ROS_TIME)
{
  axis_linear_x_ = static_cast<int>(declare_parameter<int64_t>("axis.linear_x", 1));
  axis_linear_y_ = static_cast<int>(declare_parameter<int64_t>("axis.linear_y", 0));
  axis_angular_z_ = static_cast<int>(declare_parameter<int64_t>("axis.angular_z", 3));
  axis_body_pitch_ = static_cast<int>(declare_parameter<int64_t>("axis.body_pitch", 4));
  deadman_button_ = static_cast<int>(declare_parameter<int64_t>("button.deadman", 6));
  posture_mode_button_ =
    static_cast<int>(declare_parameter<int64_t>("button.posture_mode", 4));

  scale_linear_x_ = declare_parameter<double>("scale.linear_x", scale_linear_x_);
  scale_linear_y_ = declare_parameter<double>("scale.linear_y", scale_linear_y_);
  scale_angular_z_ = declare_parameter<double>("scale.angular_z", scale_angular_z_);
  scale_body_pitch_ = declare_parameter<double>("scale.body_pitch", scale_body_pitch_);
  deadzone_ = declare_parameter<double>("deadzone", deadzone_);
  stand_height_ = declare_parameter<double>("stand_height", stand_height_);
  joy_timeout_ = declare_parameter<double>("joy_timeout", joy_timeout_);

  // Gesture buttons arrive as two parallel arrays so the mapping stays editable
  // in YAML without a custom parameter type.
  const auto indices = declare_parameter<std::vector<int64_t>>(
    "gesture.buttons", std::vector<int64_t>{0, 1, 2, 3});
  const auto ids = declare_parameter<std::vector<std::string>>(
    "gesture.ids", std::vector<std::string>{"greet", "celebrate", "thinking", "settle"});

  if (indices.size() != ids.size()) {
    RCLCPP_ERROR(
      get_logger(),
      "gesture.buttons has %zu entries but gesture.ids has %zu -- no gesture buttons bound",
      indices.size(), ids.size());
  } else {
    for (std::size_t i = 0; i < indices.size(); ++i) {
      gesture_buttons_.emplace(static_cast<int>(indices[i]), ids[i]);
      RCLCPP_INFO(
        get_logger(), "button %ld -> gesture '%s'", indices[i], ids[i].c_str());
    }
  }

  cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
  gesture_pub_ = create_publisher<pkun_msgs::msg::Gesture>("gesture", 10);
  body_pose_pub_ = create_publisher<pkun_msgs::msg::BodyPose>("body_pose", 10);

  joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
    "joy", rclcpp::QoS(10),
    std::bind(&JoyTeleopNode::on_joy, this, std::placeholders::_1));

  watchdog_ = create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&JoyTeleopNode::on_timeout_check, this));

  RCLCPP_INFO(
    get_logger(), "teleop ready -- hold button %d to walk", deadman_button_);
}

double JoyTeleopNode::axis(
  const sensor_msgs::msg::Joy & joy, int index, double scale) const
{
  if (index < 0 || index >= static_cast<int>(joy.axes.size())) {
    return 0.0;
  }
  double value = joy.axes[static_cast<std::size_t>(index)];
  if (std::abs(value) < deadzone_) {
    return 0.0;
  }
  // Rescale past the deadzone so the stick still reaches full output rather
  // than losing the deadzone fraction off the top.
  const double sign = value < 0.0 ? -1.0 : 1.0;
  value = sign * (std::abs(value) - deadzone_) / (1.0 - deadzone_);
  return value * scale;
}

bool JoyTeleopNode::pressed(const sensor_msgs::msg::Joy & joy, int index) const
{
  if (index < 0 || index >= static_cast<int>(joy.buttons.size())) {
    return false;
  }
  return joy.buttons[static_cast<std::size_t>(index)] != 0;
}

bool JoyTeleopNode::rising_edge(const sensor_msgs::msg::Joy & joy, int index)
{
  if (index < 0 || index >= static_cast<int>(joy.buttons.size())) {
    return false;
  }
  const bool now_down = joy.buttons[static_cast<std::size_t>(index)] != 0;
  const bool was_down = index < static_cast<int>(previous_buttons_.size()) &&
    previous_buttons_[static_cast<std::size_t>(index)] != 0;
  return now_down && !was_down;
}

void JoyTeleopNode::on_joy(const sensor_msgs::msg::Joy::SharedPtr msg)
{
  last_joy_stamp_ = now();

  for (const auto & [button, id] : gesture_buttons_) {
    if (rising_edge(*msg, button)) {
      pkun_msgs::msg::Gesture gesture;
      gesture.header.stamp = now();
      gesture.id = id;
      gesture.speed = 1.0f;
      gesture.loop = false;
      gesture_pub_->publish(gesture);
      RCLCPP_INFO(get_logger(), "gesture '%s'", id.c_str());
    }
  }

  const bool posture_mode = pressed(*msg, posture_mode_button_);
  deadman_held_ = pressed(*msg, deadman_button_);

  if (posture_mode) {
    // Pose the body over planted feet -- useful for aiming the head without
    // taking a step.
    pkun_msgs::msg::BodyPose pose;
    pose.header.stamp = now();
    pose.height = stand_height_ + axis(*msg, axis_linear_x_, 15.0);
    pose.x = axis(*msg, axis_body_pitch_, 12.0);
    pose.y = axis(*msg, axis_linear_y_, 12.0);
    pose.roll = 0.0;
    pose.pitch = axis(*msg, axis_body_pitch_, scale_body_pitch_);
    pose.yaw = axis(*msg, axis_angular_z_, 0.30);
    body_pose_pub_->publish(pose);

    if (!sent_stop_) {
      cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
      sent_stop_ = true;
    }
  } else if (deadman_held_) {
    geometry_msgs::msg::Twist twist;
    twist.linear.x = axis(*msg, axis_linear_x_, scale_linear_x_);
    twist.linear.y = axis(*msg, axis_linear_y_, scale_linear_y_);
    twist.angular.z = axis(*msg, axis_angular_z_, scale_angular_z_);
    cmd_vel_pub_->publish(twist);
    sent_stop_ = false;
  } else if (!sent_stop_) {
    // Deadman released: one explicit zero, then silence. Motion has its own
    // staleness timeout as the second line of defence.
    cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
    sent_stop_ = true;
    RCLCPP_INFO(get_logger(), "deadman released -- stopped");
  }

  previous_buttons_ = msg->buttons;
}

void JoyTeleopNode::on_timeout_check()
{
  if (last_joy_stamp_.nanoseconds() == 0 || sent_stop_) {
    return;
  }

  // The pad walked out of Bluetooth range or its battery died. Joy simply stops
  // arriving -- there is no disconnect event to react to, so this is the only
  // way to notice.
  const double age = (now() - last_joy_stamp_).seconds();
  if (age > joy_timeout_) {
    cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
    sent_stop_ = true;
    deadman_held_ = false;
    RCLCPP_WARN(get_logger(), "joy silent for %.1f s -- stopping", age);
  }
}

}  // namespace pkun_teleop

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pkun_teleop::JoyTeleopNode)
