#ifndef PKUN_MOTION__MOTION_NODE_HPP_
#define PKUN_MOTION__MOTION_NODE_HPP_

#include <array>
#include <deque>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "pkun_motion/leg_kinematics.hpp"
#include "pkun_msgs/msg/body_pose.hpp"
#include "pkun_msgs/msg/gesture.hpp"
#include "pkun_msgs/msg/joint_command.hpp"
#include "rclcpp/rclcpp.hpp"

namespace pkun_motion
{

/// One posture keyframe in a gesture clip.
struct PostureKey
{
  double height;
  double x;
  double y;
  double roll;
  double pitch;
  double yaw;
  double duration;   // seconds to blend from the previous key into this one
};

enum class MotionMode
{
  Idle,       ///< hold the rest stance, with a slow breathing bob
  Posture,    ///< track a commanded 6-DOF body pose, feet planted
  Walk,       ///< trot gait driven by cmd_vel
  Gesture     ///< play a posture keyframe clip to completion
};

/// Turns intent (velocity, posture, gesture) into 12 leg joint angles.
///
/// Contains no hardware access at all -- it publishes JointCommand and lets the
/// driver worry about pulses. That separation is what lets this node be tested
/// on a laptop with nothing plugged in.
class MotionNode : public rclcpp::Node
{
public:
  explicit MotionNode(const rclcpp::NodeOptions & options);

private:
  void on_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg);
  void on_body_pose(const pkun_msgs::msg::BodyPose::SharedPtr msg);
  void on_gesture(const pkun_msgs::msg::Gesture::SharedPtr msg);
  void on_tick();

  /// Feet planted, body follows `pose`. Fills `q` or returns false if unreachable.
  bool solve_posture(const PostureKey & pose, std::array<double, kNumJoints> & q);

  /// Trot gait sample at gait phase `phase` (0..1).
  bool solve_walk(double phase, std::array<double, kNumJoints> & q);

  void publish(const std::array<double, kNumJoints> & q, double duration);

  /// Built-in clip table. Returns false for an unknown gesture id.
  bool load_clip(const std::string & id, double speed, std::deque<PostureKey> & clip) const;

  LinkParams link_params_;
  JointLimits limits_;
  int knee_{+1};

  double stand_height_{110.0};
  double splay_rad_{12.0 * M_PI / 180.0};
  double publish_rate_hz_{50.0};

  // gait
  double cycle_time_{0.6};
  double step_height_{18.0};
  double duty_{0.5};
  double gait_phase_{0.0};

  // breathing idle
  double breathe_amp_{2.0};
  double breathe_period_{4.0};
  double clock_{0.0};

  std::array<Eigen::Vector3d, kNumLegs> footprint_;
  std::array<double, kNumJoints> last_q_{};
  bool have_last_q_{false};

  MotionMode mode_{MotionMode::Idle};
  geometry_msgs::msg::Twist cmd_vel_;
  rclcpp::Time last_cmd_vel_stamp_;
  PostureKey posture_target_{};

  std::deque<PostureKey> clip_;
  PostureKey clip_from_{};
  double clip_elapsed_{0.0};
  bool clip_loop_{false};
  std::string clip_id_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<pkun_msgs::msg::BodyPose>::SharedPtr body_pose_sub_;
  rclcpp::Subscription<pkun_msgs::msg::Gesture>::SharedPtr gesture_sub_;
  rclcpp::Publisher<pkun_msgs::msg::JointCommand>::SharedPtr command_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace pkun_motion

#endif  // PKUN_MOTION__MOTION_NODE_HPP_
