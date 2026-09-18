#include "pkun_servo_driver/servo_driver_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <utility>

namespace pkun_servo_driver
{
namespace
{

constexpr double kDegToRad = M_PI / 180.0;
constexpr double kRadToDeg = 180.0 / M_PI;

}  // namespace

ServoDriverNode::ServoDriverNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("servo_driver", options)
{
  declare_parameter<std::string>("i2c_bus", "/dev/i2c-1");
  declare_parameter<std::vector<int64_t>>("board_addresses", {0x40, 0x41});
  declare_parameter<double>("update_rate_hz", 50.0);
  declare_parameter<double>("state_rate_hz", 10.0);
  declare_parameter<std::vector<std::string>>("joints", std::vector<std::string>{});
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

bool ServoDriverNode::load_servo_config(std::string & error)
{
  const auto names = get_parameter("joints").as_string_array();
  if (names.empty()) {
    error = "parameter 'joints' is empty -- nothing to drive";
    return false;
  }

  servos_.clear();
  index_by_name_.clear();
  servos_.reserve(names.size());

  for (const auto & name : names) {
    const std::string prefix = "servos." + name + ".";

    ServoChannel servo;
    servo.name = name;
    servo.board = static_cast<std::size_t>(
      declare_parameter<int64_t>(prefix + "board", 0));
    servo.channel = static_cast<uint8_t>(
      declare_parameter<int64_t>(prefix + "channel", 0));
    servo.center_us = declare_parameter<double>(prefix + "center_us", 1500.0);
    servo.us_per_rad =
      declare_parameter<double>(prefix + "us_per_deg", 11.111) * kRadToDeg;
    servo.min_rad = declare_parameter<double>(prefix + "min_deg", -90.0) * kDegToRad;
    servo.max_rad = declare_parameter<double>(prefix + "max_deg", 90.0) * kDegToRad;
    servo.invert = declare_parameter<bool>(prefix + "invert", false);
    servo.trim_rad = declare_parameter<double>(prefix + "trim_deg", 0.0) * kDegToRad;
    servo.max_rate_rad_s =
      declare_parameter<double>(prefix + "max_rate_dps", 300.0) * kDegToRad;

    if (servo.board >= board_addresses_.size()) {
      error = "joint '" + name + "' references board index " +
        std::to_string(servo.board) + " but only " +
        std::to_string(board_addresses_.size()) + " boards are configured";
      return false;
    }
    if (servo.channel >= Pca9685::kNumChannels) {
      error = "joint '" + name + "' has channel " + std::to_string(servo.channel) +
        ", must be 0..15";
      return false;
    }
    if (servo.min_rad >= servo.max_rad) {
      error = "joint '" + name + "' has min_deg >= max_deg";
      return false;
    }

    // Catch two joints wired to the same pin now, rather than as a mystery
    // twitch once the robot is standing on it.
    for (const auto & other : servos_) {
      if (other.board == servo.board && other.channel == servo.channel) {
        error = "joints '" + other.name + "' and '" + name +
          "' are both mapped to board " + std::to_string(servo.board) +
          " channel " + std::to_string(servo.channel);
        return false;
      }
    }

    servo.current_rad = std::clamp(0.0, servo.min_rad, servo.max_rad);
    servo.target_rad = servo.current_rad;
    index_by_name_.emplace(name, servos_.size());
    servos_.push_back(servo);
  }

  return true;
}

ServoDriverNode::CallbackReturn ServoDriverNode::on_configure(
  const rclcpp_lifecycle::State &)
{
  i2c_bus_ = get_parameter("i2c_bus").as_string();
  board_addresses_ = get_parameter("board_addresses").as_integer_array();
  update_rate_hz_ = get_parameter("update_rate_hz").as_double();

  if (board_addresses_.empty()) {
    RCLCPP_ERROR(get_logger(), "no board_addresses configured");
    return CallbackReturn::FAILURE;
  }
  if (update_rate_hz_ <= 0.0) {
    RCLCPP_ERROR(get_logger(), "update_rate_hz must be positive");
    return CallbackReturn::FAILURE;
  }

  std::string error;
  if (!load_servo_config(error)) {
    RCLCPP_ERROR(get_logger(), "servo config rejected: %s", error.c_str());
    return CallbackReturn::FAILURE;
  }

  // This is the "initial connection to the device" step -- it lives here, in
  // the driver that owns the hardware, not in a separate node.
  boards_.clear();
  for (const auto address : board_addresses_) {
    auto board = std::make_unique<Pca9685>(i2c_bus_, static_cast<uint8_t>(address));
    if (!board->open(error)) {
      RCLCPP_ERROR(
        get_logger(), "PCA9685 at 0x%02x on %s did not respond: %s",
        static_cast<unsigned>(address), i2c_bus_.c_str(), error.c_str());
      boards_.clear();
      return CallbackReturn::FAILURE;
    }
    if (!board->set_frequency(update_rate_hz_, error)) {
      RCLCPP_ERROR(get_logger(), "could not set PWM frequency: %s", error.c_str());
      boards_.clear();
      return CallbackReturn::FAILURE;
    }
    RCLCPP_INFO(
      get_logger(), "PCA9685 0x%02x ready, PWM %.2f Hz",
      static_cast<unsigned>(address), board->frequency());
    boards_.push_back(std::move(board));
  }

  state_pub_ = create_publisher<pkun_msgs::msg::ServoState>("~/servo_state", 10);

  // Subscribed while inactive too: commands arriving early are absorbed into
  // the target and simply not written out until activation.
  command_sub_ = create_subscription<pkun_msgs::msg::JointCommand>(
    "joint_command", rclcpp::QoS(10),
    std::bind(&ServoDriverNode::on_joint_command, this, std::placeholders::_1));

  set_trim_srv_ = create_service<pkun_msgs::srv::SetTrim>(
    "~/set_trim",
    std::bind(
      &ServoDriverNode::handle_set_trim, this,
      std::placeholders::_1, std::placeholders::_2));
  set_torque_srv_ = create_service<pkun_msgs::srv::SetTorque>(
    "~/set_torque",
    std::bind(
      &ServoDriverNode::handle_set_torque, this,
      std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(
    get_logger(), "configured %zu servos across %zu board(s)",
    servos_.size(), boards_.size());
  return CallbackReturn::SUCCESS;
}

ServoDriverNode::CallbackReturn ServoDriverNode::on_activate(
  const rclcpp_lifecycle::State & state)
{
  LifecycleNode::on_activate(state);

  torque_enabled_ = true;

  // Write the current pose once before the timer starts, so the first tick has
  // somewhere sane to interpolate from instead of snapping.
  for (auto & servo : servos_) {
    servo.target_rad = servo.current_rad;
    servo.step_rad = servo.max_rate_rad_s / update_rate_hz_;
    write_servo(servo, servo.current_rad);
  }

  const auto period = std::chrono::duration<double>(1.0 / update_rate_hz_);
  control_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&ServoDriverNode::on_control_tick, this));

  const double state_hz = std::max(1.0, get_parameter("state_rate_hz").as_double());
  state_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / state_hz)),
    [this]() {
      pkun_msgs::msg::ServoState msg;
      msg.header.stamp = now();
      msg.name.reserve(servos_.size());
      msg.position.reserve(servos_.size());
      msg.pulse_us.reserve(servos_.size());
      for (const auto & servo : servos_) {
        msg.name.push_back(servo.name);
        msg.position.push_back(servo.current_rad);
        msg.pulse_us.push_back(servo.last_pulse_us);
      }
      msg.torque_enabled = torque_enabled_;
      msg.boards_online = static_cast<uint8_t>(boards_.size());
      state_pub_->publish(msg);
    });

  RCLCPP_INFO(get_logger(), "active -- driving %zu servos", servos_.size());
  return CallbackReturn::SUCCESS;
}

ServoDriverNode::CallbackReturn ServoDriverNode::on_deactivate(
  const rclcpp_lifecycle::State & state)
{
  control_timer_.reset();
  state_timer_.reset();
  release_all();
  LifecycleNode::on_deactivate(state);
  RCLCPP_INFO(get_logger(), "inactive -- servos released");
  return CallbackReturn::SUCCESS;
}

ServoDriverNode::CallbackReturn ServoDriverNode::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  control_timer_.reset();
  state_timer_.reset();
  command_sub_.reset();
  set_trim_srv_.reset();
  set_torque_srv_.reset();
  state_pub_.reset();
  boards_.clear();      // destructor releases every channel and closes the bus
  servos_.clear();
  index_by_name_.clear();
  return CallbackReturn::SUCCESS;
}

ServoDriverNode::CallbackReturn ServoDriverNode::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  control_timer_.reset();
  state_timer_.reset();
  release_all();
  boards_.clear();
  return CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// command handling
// ---------------------------------------------------------------------------

void ServoDriverNode::on_joint_command(
  const pkun_msgs::msg::JointCommand::SharedPtr msg)
{
  if (msg->name.size() != msg->position.size()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "joint_command has %zu names but %zu positions -- ignoring",
      msg->name.size(), msg->position.size());
    return;
  }

  for (std::size_t i = 0; i < msg->name.size(); ++i) {
    const auto it = index_by_name_.find(msg->name[i]);
    if (it == index_by_name_.end()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "unknown joint '%s' in joint_command", msg->name[i].c_str());
      continue;
    }

    auto & servo = servos_[it->second];
    const double requested = msg->position[i];
    const double clamped = std::clamp(requested, servo.min_rad, servo.max_rad);
    if (std::abs(clamped - requested) > 1e-6) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "joint '%s' commanded to %.1f deg, clamped to %.1f deg",
        servo.name.c_str(), requested * kRadToDeg, clamped * kRadToDeg);
    }
    servo.target_rad = clamped;

    // A duration turns into a per-tick step. It can only ever slow the servo
    // down: max_rate_dps is a torque/heat limit and is never exceeded.
    double rate = servo.max_rate_rad_s;
    if (msg->duration > 1e-3) {
      const double needed = std::abs(clamped - servo.current_rad) / msg->duration;
      rate = std::min(rate, needed);
    }
    servo.step_rad = std::max(rate, 1e-6) / update_rate_hz_;
  }
}

void ServoDriverNode::on_control_tick()
{
  if (!torque_enabled_) {
    return;
  }

  for (auto & servo : servos_) {
    if (!servo.enabled) {
      continue;
    }
    const double error = servo.target_rad - servo.current_rad;
    const double step = std::clamp(error, -servo.step_rad, servo.step_rad);
    write_servo(servo, servo.current_rad + step);
  }
}

double ServoDriverNode::pulse_for(const ServoChannel & servo, double angle_rad) const
{
  const double sign = servo.invert ? -1.0 : 1.0;
  return servo.center_us + sign * (angle_rad + servo.trim_rad) * servo.us_per_rad;
}

void ServoDriverNode::write_servo(ServoChannel & servo, double angle_rad)
{
  const double clamped = std::clamp(angle_rad, servo.min_rad, servo.max_rad);
  const double pulse = pulse_for(servo, clamped);

  std::string error;
  if (!boards_[servo.board]->set_pulse_us(servo.channel, pulse, error)) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "write failed for '%s': %s", servo.name.c_str(), error.c_str());
    return;
  }
  servo.current_rad = clamped;
  servo.last_pulse_us = pulse;
}

void ServoDriverNode::release_all()
{
  std::string error;
  for (auto & board : boards_) {
    if (!board->all_off(error)) {
      RCLCPP_WARN(get_logger(), "failed to release a board: %s", error.c_str());
    }
  }
  for (auto & servo : servos_) {
    servo.last_pulse_us = 0.0;
  }
}

// ---------------------------------------------------------------------------
// services
// ---------------------------------------------------------------------------

void ServoDriverNode::handle_set_trim(
  const std::shared_ptr<pkun_msgs::srv::SetTrim::Request> request,
  std::shared_ptr<pkun_msgs::srv::SetTrim::Response> response)
{
  std::vector<std::size_t> targets;
  if (request->name.empty()) {
    targets.resize(servos_.size());
    for (std::size_t i = 0; i < servos_.size(); ++i) {targets[i] = i;}
  } else {
    const auto it = index_by_name_.find(request->name);
    if (it == index_by_name_.end()) {
      response->success = false;
      response->message = "unknown joint '" + request->name + "'";
      return;
    }
    targets.push_back(it->second);
  }

  for (const auto index : targets) {
    servos_[index].trim_rad = request->trim;
    if (request->persist) {
      set_parameter(
        rclcpp::Parameter(
          "servos." + servos_[index].name + ".trim_deg", request->trim * kRadToDeg));
    }
  }

  response->success = true;
  response->message = "trim set to " + std::to_string(request->trim * kRadToDeg) +
    " deg on " + std::to_string(targets.size()) + " joint(s)";
  RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
}

void ServoDriverNode::handle_set_torque(
  const std::shared_ptr<pkun_msgs::srv::SetTorque::Request> request,
  std::shared_ptr<pkun_msgs::srv::SetTorque::Response> response)
{
  if (request->name.empty()) {
    torque_enabled_ = request->enable;
    for (auto & servo : servos_) {servo.enabled = request->enable;}
    if (!request->enable) {
      release_all();
    } else {
      for (auto & servo : servos_) {
        // Resume from where the horn already is, not from a stale target that
        // would make the robot snap on re-enable.
        servo.target_rad = servo.current_rad;
        write_servo(servo, servo.current_rad);
      }
    }
    response->success = true;
    response->message = request->enable ? "all servos enabled" : "all servos released";
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    return;
  }

  for (const auto & name : request->name) {
    const auto it = index_by_name_.find(name);
    if (it == index_by_name_.end()) {
      response->success = false;
      response->message = "unknown joint '" + name + "'";
      return;
    }
    auto & servo = servos_[it->second];
    servo.enabled = request->enable;
    if (request->enable) {
      servo.target_rad = servo.current_rad;
      write_servo(servo, servo.current_rad);
    } else {
      std::string error;
      boards_[servo.board]->set_pulse_us(servo.channel, 0.0, error);
      servo.last_pulse_us = 0.0;
    }
  }

  response->success = true;
  response->message = std::to_string(request->name.size()) +
    (request->enable ? " joint(s) enabled" : " joint(s) released");
  return;
}

}  // namespace pkun_servo_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pkun_servo_driver::ServoDriverNode)
