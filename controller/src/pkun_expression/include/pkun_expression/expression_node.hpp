#ifndef PKUN_EXPRESSION__EXPRESSION_NODE_HPP_
#define PKUN_EXPRESSION__EXPRESSION_NODE_HPP_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pkun_expression/display.hpp"
#include "pkun_expression/sound_player.hpp"
#include "pkun_msgs/msg/expression.hpp"
#include "pkun_msgs/msg/gesture.hpp"
#include "pkun_msgs/msg/joint_command.hpp"
#include "rclcpp/rclcpp.hpp"

namespace pkun_expression
{

/// One expression keyframe: the six servo channels plus the face channels.
struct ExpressionKey
{
  double head_pitch{0.0};   // degrees
  double head_yaw{0.0};
  double head_roll{0.0};
  double ear_l{0.0};        // -1..1
  double ear_r{0.0};
  double eye{1.0};          // 0..1
  double tail{0.0};         // degrees
  uint8_t led{1};
  double led_v{0.35};
  double sound{0.0};
  double duration{0.3};     // seconds to blend into this key
};

/// Renders the robot's face: LCD, speaker, and the six expression servos.
///
/// Note on I2C: the servo driver owns the PCA9685 boards and this node owns the
/// display. They are different chips at different addresses, and each opens its
/// own file descriptor -- the kernel i2c-dev layer serialises the transfers. The
/// rule that matters is that no two nodes drive the same device, and that holds:
/// the head and ear servos are commanded through JointCommand like every other
/// joint, never written to directly from here.
class ExpressionNode : public rclcpp::Node
{
public:
  explicit ExpressionNode(const rclcpp::NodeOptions & options);
  ~ExpressionNode() override;

private:
  void on_expression(const pkun_msgs::msg::Expression::SharedPtr msg);
  void on_gesture(const pkun_msgs::msg::Gesture::SharedPtr msg);
  void on_tick();

  void publish_servos(const ExpressionKey & key);
  void render(const ExpressionKey & key);
  bool load_clip(const std::string & id, std::deque<ExpressionKey> & clip) const;

  std::unique_ptr<Display> display_;
  std::unique_ptr<SoundPlayer> sound_;

  std::string display_kind_{"console"};
  double update_rate_hz_{50.0};
  double render_rate_hz_{20.0};
  double blink_period_{4.5};
  double blink_duration_{0.16};

  std::map<uint8_t, std::string> led_names_;
  std::map<std::string, std::string> gesture_sounds_;

  ExpressionKey current_{};
  ExpressionKey target_{};
  ExpressionKey blend_from_{};
  double blend_elapsed_{0.0};
  double clock_{0.0};
  double next_blink_{4.5};
  double render_accumulator_{0.0};

  std::deque<ExpressionKey> clip_;

  rclcpp::Subscription<pkun_msgs::msg::Expression>::SharedPtr expression_sub_;
  rclcpp::Subscription<pkun_msgs::msg::Gesture>::SharedPtr gesture_sub_;
  rclcpp::Publisher<pkun_msgs::msg::JointCommand>::SharedPtr command_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace pkun_expression

#endif  // PKUN_EXPRESSION__EXPRESSION_NODE_HPP_
