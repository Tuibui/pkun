#include "pkun_expression/expression_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace pkun_expression
{
namespace
{

constexpr double kDegToRad = M_PI / 180.0;

/// Ear travel: gestures.py scales 1 unit of ear to 45 degrees of servo.
constexpr double kEarScaleDeg = 45.0;

double smoothstep(double u)
{
  u = std::clamp(u, 0.0, 1.0);
  return u * u * (3.0 - 2.0 * u);
}

double lerp(double from, double to, double s)
{
  return from + (to - from) * s;
}

ExpressionKey blend(const ExpressionKey & from, const ExpressionKey & to, double u)
{
  const double s = smoothstep(u);
  ExpressionKey out = to;
  out.head_pitch = lerp(from.head_pitch, to.head_pitch, s);
  out.head_yaw = lerp(from.head_yaw, to.head_yaw, s);
  out.head_roll = lerp(from.head_roll, to.head_roll, s);
  out.ear_l = lerp(from.ear_l, to.ear_l, s);
  out.ear_r = lerp(from.ear_r, to.ear_r, s);
  out.eye = lerp(from.eye, to.eye, s);
  out.tail = lerp(from.tail, to.tail, s);
  out.led_v = lerp(from.led_v, to.led_v, s);
  out.sound = lerp(from.sound, to.sound, s);
  // led is an index into a palette, so it steps -- interpolating it would walk
  // through unrelated colours on the way.
  out.led = u >= 0.5 ? to.led : from.led;
  return out;
}

}  // namespace

ExpressionNode::ExpressionNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("expression", options)
{
  display_kind_ = declare_parameter<std::string>("display.backend", "console");
  const auto display_bus = declare_parameter<std::string>("display.i2c_bus", "/dev/i2c-1");
  const auto display_address =
    static_cast<uint8_t>(declare_parameter<int64_t>("display.address", 0x3C));

  update_rate_hz_ = declare_parameter<double>("update_rate_hz", update_rate_hz_);
  render_rate_hz_ = declare_parameter<double>("render_rate_hz", render_rate_hz_);
  blink_period_ = declare_parameter<double>("blink.period", blink_period_);
  blink_duration_ = declare_parameter<double>("blink.duration", blink_duration_);

  const auto sound_command = declare_parameter<std::string>("sound.command", "aplay");
  const auto sound_args = declare_parameter<std::vector<std::string>>(
    "sound.args", std::vector<std::string>{"-q"});
  const auto sound_dir = declare_parameter<std::string>("sound.dir", "");

  const auto led_names = declare_parameter<std::vector<std::string>>(
    "led_names",
    std::vector<std::string>{
    "off", "rest", "low", "soon", "urgent", "good", "neutral",
    "warn", "listen", "think", "warm", "power", "focus", "speak"});
  for (std::size_t i = 0; i < led_names.size(); ++i) {
    led_names_.emplace(static_cast<uint8_t>(i), led_names[i]);
  }

  const auto sound_ids = declare_parameter<std::vector<std::string>>(
    "sound.gesture_ids", std::vector<std::string>{});
  const auto sound_files = declare_parameter<std::vector<std::string>>(
    "sound.gesture_files", std::vector<std::string>{});
  if (sound_ids.size() != sound_files.size()) {
    RCLCPP_ERROR(
      get_logger(),
      "sound.gesture_ids has %zu entries but sound.gesture_files has %zu -- no cues bound",
      sound_ids.size(), sound_files.size());
  } else {
    for (std::size_t i = 0; i < sound_ids.size(); ++i) {
      gesture_sounds_.emplace(sound_ids[i], sound_files[i]);
    }
  }

  std::string error;
  display_ = make_display(display_kind_, display_bus, display_address, error);
  if (!display_) {
    RCLCPP_ERROR(get_logger(), "%s -- falling back to console", error.c_str());
    display_kind_ = "console";
    display_ = std::make_unique<ConsoleDisplay>();
  }
  if (!display_->open(error)) {
    // A missing panel must not take the robot down: the servos and the gait are
    // still perfectly usable without a face.
    RCLCPP_ERROR(
      get_logger(), "display '%s' failed to open (%s) -- falling back to console",
      display_kind_.c_str(), error.c_str());
    display_ = std::make_unique<ConsoleDisplay>();
    display_kind_ = "console";
    display_->open(error);
  }

  sound_ = std::make_unique<SoundPlayer>(sound_command, sound_args, sound_dir);

  command_pub_ = create_publisher<pkun_msgs::msg::JointCommand>("joint_command", 10);
  expression_sub_ = create_subscription<pkun_msgs::msg::Expression>(
    "expression", rclcpp::QoS(10),
    std::bind(&ExpressionNode::on_expression, this, std::placeholders::_1));
  gesture_sub_ = create_subscription<pkun_msgs::msg::Gesture>(
    "gesture", rclcpp::QoS(10),
    std::bind(&ExpressionNode::on_gesture, this, std::placeholders::_1));

  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / update_rate_hz_)),
    std::bind(&ExpressionNode::on_tick, this));

  RCLCPP_INFO(get_logger(), "expression ready, display backend '%s'", display_kind_.c_str());
}

ExpressionNode::~ExpressionNode()
{
  if (display_) {
    display_->close();
  }
  if (sound_) {
    sound_->stop();
  }
}

// ---------------------------------------------------------------------------

void ExpressionNode::on_expression(const pkun_msgs::msg::Expression::SharedPtr msg)
{
  // A direct expression command overrides whatever clip is running.
  clip_.clear();
  blend_from_ = current_;
  blend_elapsed_ = 0.0;

  target_ = ExpressionKey{
    msg->head_pitch, msg->head_yaw, msg->head_roll,
    msg->ear_l, msg->ear_r, msg->eye, msg->tail,
    msg->led, msg->led_v, msg->sound, 0.15};
}

void ExpressionNode::on_gesture(const pkun_msgs::msg::Gesture::SharedPtr msg)
{
  std::deque<ExpressionKey> clip;
  if (!load_clip(msg->id, clip)) {
    RCLCPP_DEBUG(get_logger(), "no expression track for gesture '%s'", msg->id.c_str());
    return;
  }

  const double speed = msg->speed > 0.01f ? msg->speed : 1.0f;
  for (auto & key : clip) {
    key.duration /= speed;
  }

  clip_ = std::move(clip);
  blend_from_ = current_;
  blend_elapsed_ = 0.0;
  target_ = clip_.front();

  const auto cue = gesture_sounds_.find(msg->id);
  if (cue != gesture_sounds_.end()) {
    std::string error;
    if (!sound_->play(cue->second, error)) {
      RCLCPP_WARN(get_logger(), "sound cue for '%s' failed: %s", msg->id.c_str(), error.c_str());
    }
  }
}

void ExpressionNode::on_tick()
{
  const double dt = 1.0 / update_rate_hz_;
  clock_ += dt;
  blend_elapsed_ += dt;
  sound_->poll();

  const double duration = std::max(1e-3, target_.duration);
  const double u = blend_elapsed_ / duration;
  current_ = blend(blend_from_, target_, u);

  if (u >= 1.0) {
    blend_from_ = target_;
    blend_elapsed_ = 0.0;
    if (!clip_.empty()) {
      clip_.pop_front();
      if (!clip_.empty()) {
        target_ = clip_.front();
      }
    }
  }

  // Idle blink, layered on top of whatever the clip asked for. Without this the
  // face reads as frozen between gestures.
  if (clip_.empty() && clock_ >= next_blink_) {
    const double since = clock_ - next_blink_;
    if (since < blink_duration_) {
      const double phase = since / blink_duration_;
      current_.eye *= std::abs(std::cos(M_PI * phase));
    } else {
      // Irregular spacing: a metronome blink looks more mechanical than none.
      next_blink_ = clock_ + blink_period_ * (0.6 + 0.8 * (std::sin(clock_) * 0.5 + 0.5));
    }
  }

  publish_servos(current_);

  // The panel is far slower than the control loop; redrawing at 50 Hz would
  // just block on I2C.
  render_accumulator_ += dt;
  if (render_accumulator_ >= 1.0 / render_rate_hz_) {
    render_accumulator_ = 0.0;
    render(current_);
  }
}

void ExpressionNode::publish_servos(const ExpressionKey & key)
{
  pkun_msgs::msg::JointCommand msg;
  msg.header.stamp = now();
  msg.name = {"head_pitch", "head_yaw", "head_roll", "ear_l", "ear_r", "tail"};
  msg.position = {
    key.head_pitch * kDegToRad,
    key.head_yaw * kDegToRad,
    key.head_roll * kDegToRad,
    key.ear_l * kEarScaleDeg * kDegToRad,
    key.ear_r * kEarScaleDeg * kDegToRad,
    key.tail * kDegToRad};
  msg.duration = 1.0 / update_rate_hz_;
  command_pub_->publish(msg);
}

void ExpressionNode::render(const ExpressionKey & key)
{
  FaceState face;
  face.eye = key.eye;
  face.led = key.led;
  face.led_v = key.led_v;
  const auto name = led_names_.find(key.led);
  face.led_name = name != led_names_.end() ? name->second : "?";

  std::string error;
  if (!display_->render(face, error)) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "display render failed: %s", error.c_str());
    return;
  }

  if (display_kind_ == "console") {
    const auto * console = dynamic_cast<ConsoleDisplay *>(display_.get());
    if (console != nullptr) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000, "%s", console->last_frame().c_str());
    }
  }
}

// ---------------------------------------------------------------------------
// clip table -- the head/ear/eye/LED/sound tracks of the gestures.py clips
// ---------------------------------------------------------------------------

bool ExpressionNode::load_clip(
  const std::string & id, std::deque<ExpressionKey> & clip) const
{
  // LED indices match the constants in Expression.msg.
  constexpr uint8_t kRest = 1, kLow = 2, kSoon = 3, kUrgent = 4, kGood = 5;
  constexpr uint8_t kNeutral = 6, kWarn = 7, kListen = 8, kThink = 9;
  constexpr uint8_t kWarm = 10, kPower = 11, kFocus = 12, kSpeak = 13;

  auto key = [&clip](
    double pitch, double yaw, double roll, double ears, double eye,
    double tail, uint8_t led, double led_v, double sound, double seconds) {
      clip.push_back(
        ExpressionKey{pitch, yaw, roll, ears, ears, eye, tail, led, led_v, sound, seconds});
    };

  clip.clear();

  if (id == "greet") {
    key(12.0, 0.0, 0.0, 0.8, 1.0, 25.0, kGood, 0.8, 0.6, 0.30);
    key(-6.0, 0.0, 0.0, 0.6, 0.3, -25.0, kGood, 0.8, 0.4, 0.25);
    key(0.0, 0.0, 0.0, 0.4, 1.0, 20.0, kRest, 0.5, 0.0, 0.30);
  } else if (id == "celebrate") {
    key(18.0, 0.0, 0.0, 1.0, 1.0, 40.0, kGood, 1.0, 0.9, 0.20);
    key(10.0, 20.0, 8.0, 1.0, 1.0, -40.0, kGood, 1.0, 0.7, 0.18);
    key(10.0, -20.0, -8.0, 1.0, 1.0, 40.0, kGood, 1.0, 0.7, 0.18);
    key(0.0, 0.0, 0.0, 0.4, 1.0, 0.0, kRest, 0.5, 0.0, 0.30);
  } else if (id == "thinking") {
    key(-6.0, 12.0, 14.0, -0.3, 0.55, 0.0, kThink, 0.7, 0.0, 0.50);
    key(-6.0, -12.0, -14.0, -0.3, 0.55, 0.0, kThink, 0.7, 0.0, 0.55);
    key(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, kRest, 0.5, 0.0, 0.40);
  } else if (id == "listening") {
    key(6.0, 18.0, 0.0, 1.0, 1.0, 10.0, kListen, 0.85, 0.0, 0.35);
  } else if (id == "speaking") {
    key(4.0, 0.0, 0.0, 0.5, 1.0, 8.0, kSpeak, 0.8, 0.7, 0.25);
    key(-2.0, 0.0, 0.0, 0.5, 0.9, -8.0, kSpeak, 0.6, 0.5, 0.25);
  } else if (id == "confirm_done") {
    key(-10.0, 0.0, 0.0, 0.7, 0.4, 30.0, kGood, 1.0, 0.5, 0.15);
    key(8.0, 0.0, 0.0, 0.7, 1.0, -30.0, kGood, 1.0, 0.0, 0.15);
    key(0.0, 0.0, 0.0, 0.3, 1.0, 0.0, kRest, 0.5, 0.0, 0.25);
  } else if (id == "didnt_get_it") {
    key(-4.0, 0.0, 16.0, -0.6, 0.6, 0.0, kWarn, 0.7, 0.4, 0.28);
    key(-4.0, 0.0, -16.0, -0.6, 0.6, 0.0, kWarn, 0.7, 0.0, 0.28);
    key(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, kNeutral, 0.5, 0.0, 0.30);
  } else if (id == "notice_low") {
    key(8.0, 0.0, 0.0, 0.6, 1.0, 15.0, kLow, 0.6, 0.3, 0.30);
    key(0.0, 0.0, 0.0, 0.2, 1.0, 0.0, kRest, 0.5, 0.0, 0.35);
  } else if (id == "notice_soon") {
    key(10.0, 0.0, 0.0, 0.8, 1.0, 20.0, kSoon, 0.8, 0.5, 0.28);
    key(0.0, 0.0, 0.0, 0.3, 1.0, 0.0, kRest, 0.5, 0.0, 0.32);
  } else if (id == "notice_urgent") {
    key(16.0, 0.0, 0.0, 1.0, 1.0, 40.0, kUrgent, 1.0, 1.0, 0.12);
    key(-8.0, 0.0, 0.0, 1.0, 1.0, -40.0, kUrgent, 0.3, 0.8, 0.12);
    key(16.0, 0.0, 0.0, 1.0, 1.0, 40.0, kUrgent, 1.0, 1.0, 0.12);
    key(0.0, 0.0, 0.0, 0.4, 1.0, 0.0, kWarn, 0.6, 0.0, 0.25);
  } else if (id == "settle" || id == "home_reset") {
    key(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, kRest, 0.4, 0.0, 0.55);
  } else if (id == "focus_with_you") {
    key(6.0, 22.0, 0.0, 0.9, 1.0, 5.0, kFocus, 0.9, 0.0, 0.45);
  } else if (id == "comfort") {
    key(-8.0, 8.0, 6.0, -0.4, 0.5, 8.0, kWarm, 0.6, 0.3, 0.70);
    key(-4.0, 0.0, 0.0, -0.2, 0.7, 0.0, kWarm, 0.5, 0.0, 0.60);
  } else if (id == "idle_breathe") {
    key(2.0, 0.0, 0.0, 0.0, 1.0, 3.0, kRest, 0.35, 0.0, 1.00);
    key(-2.0, 0.0, 0.0, 0.0, 1.0, -3.0, kRest, 0.30, 0.0, 1.00);
  } else if (id == "doze_off") {
    key(-6.0, 0.0, 0.0, -0.5, 0.5, 0.0, kRest, 0.25, 0.0, 0.80);
    key(-12.0, 0.0, 0.0, -0.9, 0.1, 0.0, kRest, 0.15, 0.0, 1.00);
  } else if (id == "sleep_mode") {
    key(-14.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0, 0.0, 0.0, 1.20);
  } else if (id == "wake_up") {
    key(-14.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0, 0.0, 0.0, 0.40);
    key(10.0, 0.0, 0.0, 0.9, 1.0, 20.0, kGood, 0.9, 0.5, 0.50);
    key(0.0, 0.0, 0.0, 0.2, 1.0, 0.0, kRest, 0.5, 0.0, 0.40);
  } else if (id == "low_power") {
    key(-8.0, 0.0, 0.0, -0.7, 0.4, 0.0, kPower, 0.5, 0.4, 0.90);
  } else if (id == "error_state") {
    key(0.0, 0.0, 10.0, -0.8, 0.3, 0.0, kWarn, 1.0, 0.8, 0.12);
    key(0.0, 0.0, -10.0, -0.8, 0.3, 0.0, kWarn, 0.2, 0.0, 0.12);
    key(0.0, 0.0, 0.0, -0.4, 0.7, 0.0, kWarn, 0.6, 0.0, 0.25);
  } else {
    return false;
  }

  return true;
}

}  // namespace pkun_expression

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pkun_expression::ExpressionNode)
