#ifndef PKUN_SERVO_DRIVER__SERVO_DRIVER_NODE_HPP_
#define PKUN_SERVO_DRIVER__SERVO_DRIVER_NODE_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pkun_msgs/msg/joint_command.hpp"
#include "pkun_msgs/msg/servo_state.hpp"
#include "pkun_msgs/srv/set_torque.hpp"
#include "pkun_msgs/srv/set_trim.hpp"
#include "pkun_servo_driver/pca9685.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"

namespace pkun_servo_driver
{

/// Calibration and live state for one servo channel.
struct ServoChannel
{
  std::string name;
  std::size_t board{0};        // index into the configured board address list
  uint8_t channel{0};          // 0..15 on that board

  // Hobby servo calibration. Angle -> pulse is linear, anchored at the pose the
  // horn takes at 0 rad, so trimming the zero never rescales the travel.
  double center_us{1500.0};
  double us_per_rad{636.62};   // 1000 us per 90 deg is typical for a 180 deg servo
  double min_rad{-1.5708};
  double max_rad{1.5708};
  bool invert{false};
  double trim_rad{0.0};

  double max_rate_rad_s{5.236};  // 300 deg/s, the planning rate used in walking_gait.py

  // Live state.
  double current_rad{0.0};     // what we last wrote
  double target_rad{0.0};      // where we were told to go
  double step_rad{0.0};        // per-tick allowance toward the target
  double last_pulse_us{0.0};
  bool enabled{true};
};

/// Owns every servo on the robot and is the only place that touches I2C.
///
/// Lifecycle node on purpose: opening the bus and probing the boards belongs in
/// on_configure(), so a wiring fault fails one clean transition instead of
/// throwing from a constructor or getting retried forever in a timer.
class ServoDriverNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit ServoDriverNode(const rclcpp::NodeOptions & options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

private:
  bool load_servo_config(std::string & error);
  void on_joint_command(const pkun_msgs::msg::JointCommand::SharedPtr msg);
  void on_control_tick();
  void handle_set_trim(
    const std::shared_ptr<pkun_msgs::srv::SetTrim::Request> request,
    std::shared_ptr<pkun_msgs::srv::SetTrim::Response> response);
  void handle_set_torque(
    const std::shared_ptr<pkun_msgs::srv::SetTorque::Request> request,
    std::shared_ptr<pkun_msgs::srv::SetTorque::Response> response);

  double pulse_for(const ServoChannel & servo, double angle_rad) const;
  void write_servo(ServoChannel & servo, double angle_rad);
  void release_all();

  std::vector<std::unique_ptr<Pca9685>> boards_;
  std::vector<ServoChannel> servos_;
  std::unordered_map<std::string, std::size_t> index_by_name_;

  std::string i2c_bus_;
  std::vector<int64_t> board_addresses_;
  double update_rate_hz_{50.0};
  bool torque_enabled_{true};

  rclcpp::Subscription<pkun_msgs::msg::JointCommand>::SharedPtr command_sub_;
  rclcpp_lifecycle::LifecyclePublisher<pkun_msgs::msg::ServoState>::SharedPtr state_pub_;
  rclcpp::Service<pkun_msgs::srv::SetTrim>::SharedPtr set_trim_srv_;
  rclcpp::Service<pkun_msgs::srv::SetTorque>::SharedPtr set_torque_srv_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::TimerBase::SharedPtr state_timer_;
};

}  // namespace pkun_servo_driver

#endif  // PKUN_SERVO_DRIVER__SERVO_DRIVER_NODE_HPP_
