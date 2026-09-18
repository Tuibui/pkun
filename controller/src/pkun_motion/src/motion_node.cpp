#include "pkun_motion/motion_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace pkun_motion
{
namespace
{

constexpr double kDegToRad = M_PI / 180.0;

/// Smoothstep, the same easing sample_keys() uses in gestures.py. Zero velocity
/// at both ends, which is what keeps the servos from clacking at keyframes.
double smoothstep(double u)
{
  u = std::clamp(u, 0.0, 1.0);
  return u * u * (3.0 - 2.0 * u);
}

PostureKey blend(const PostureKey & from, const PostureKey & to, double u)
{
  const double s = smoothstep(u);
  PostureKey out = to;
  out.height = from.height + (to.height - from.height) * s;
  out.x = from.x + (to.x - from.x) * s;
  out.y = from.y + (to.y - from.y) * s;
  out.roll = from.roll + (to.roll - from.roll) * s;
  out.pitch = from.pitch + (to.pitch - from.pitch) * s;
  out.yaw = from.yaw + (to.yaw - from.yaw) * s;
  return out;
}

}  // namespace

MotionNode::MotionNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("motion", options), last_cmd_vel_stamp_(0, 0, RCL_ROS_TIME)
{
  link_params_.a = declare_parameter<double>("link.a", link_params_.a);
  link_params_.b = declare_parameter<double>("link.b", link_params_.b);
  link_params_.c = declare_parameter<double>("link.c", link_params_.c);
  link_params_.h = declare_parameter<double>("link.h", link_params_.h);
  link_params_.d1 = declare_parameter<double>("link.d1", link_params_.d1);
  link_params_.d2 = declare_parameter<double>("link.d2", link_params_.d2);
  link_params_.l1 = declare_parameter<double>("link.l1", link_params_.l1);
  link_params_.l2 = declare_parameter<double>("link.l2", link_params_.l2);

  stand_height_ = declare_parameter<double>("stand_height", stand_height_);
  splay_rad_ = declare_parameter<double>("splay_deg", 12.0) * kDegToRad;
  publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", publish_rate_hz_);
  knee_ = static_cast<int>(declare_parameter<int64_t>("knee", 1));

  cycle_time_ = declare_parameter<double>("gait.cycle_time", cycle_time_);
  step_height_ = declare_parameter<double>("gait.step_height", step_height_);
  duty_ = declare_parameter<double>("gait.duty", duty_);

  breathe_amp_ = declare_parameter<double>("idle.breathe_amp", breathe_amp_);
  breathe_period_ = declare_parameter<double>("idle.breathe_period", breathe_period_);

  limits_.abduction_min = declare_parameter<double>("limits.abduction_min_deg", -55.0) * kDegToRad;
  limits_.abduction_max = declare_parameter<double>("limits.abduction_max_deg", 55.0) * kDegToRad;
  limits_.hip_pitch_min = declare_parameter<double>("limits.hip_pitch_min_deg", -90.0) * kDegToRad;
  limits_.hip_pitch_max = declare_parameter<double>("limits.hip_pitch_max_deg", 90.0) * kDegToRad;
  limits_.knee_min = declare_parameter<double>("limits.knee_min_deg", 0.0) * kDegToRad;
  limits_.knee_max = declare_parameter<double>("limits.knee_max_deg", 150.0) * kDegToRad;

  footprint_ = footprint(link_params_, stand_height_, splay_rad_);
  posture_target_ = PostureKey{stand_height_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  command_pub_ = create_publisher<pkun_msgs::msg::JointCommand>("joint_command", 10);

  cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", rclcpp::QoS(10),
    std::bind(&MotionNode::on_cmd_vel, this, std::placeholders::_1));
  body_pose_sub_ = create_subscription<pkun_msgs::msg::BodyPose>(
    "body_pose", rclcpp::QoS(10),
    std::bind(&MotionNode::on_body_pose, this, std::placeholders::_1));
  gesture_sub_ = create_subscription<pkun_msgs::msg::Gesture>(
    "gesture", rclcpp::QoS(10),
    std::bind(&MotionNode::on_gesture, this, std::placeholders::_1));

  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / publish_rate_hz_)),
    std::bind(&MotionNode::on_tick, this));

  RCLCPP_INFO(
    get_logger(), "motion ready: stand %.0f mm, splay %.0f deg, %.0f Hz",
    stand_height_, splay_rad_ / kDegToRad, publish_rate_hz_);
}

// ---------------------------------------------------------------------------
// inputs
// ---------------------------------------------------------------------------

void MotionNode::on_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  cmd_vel_ = *msg;
  last_cmd_vel_stamp_ = now();

  const bool moving = std::abs(msg->linear.x) > 1e-3 ||
    std::abs(msg->linear.y) > 1e-3 ||
    std::abs(msg->angular.z) > 1e-3;

  // A gesture owns the body until it finishes; velocity does not interrupt it.
  if (mode_ == MotionMode::Gesture) {
    return;
  }
  if (moving) {
    mode_ = MotionMode::Walk;
  } else if (mode_ == MotionMode::Walk) {
    mode_ = MotionMode::Idle;
    gait_phase_ = 0.0;
  }
}

void MotionNode::on_body_pose(const pkun_msgs::msg::BodyPose::SharedPtr msg)
{
  posture_target_ = PostureKey{
    msg->height, msg->x, msg->y, msg->roll, msg->pitch, msg->yaw, 0.0};
  if (mode_ != MotionMode::Gesture) {
    mode_ = MotionMode::Posture;
  }
}

void MotionNode::on_gesture(const pkun_msgs::msg::Gesture::SharedPtr msg)
{
  std::deque<PostureKey> clip;
  if (!load_clip(msg->id, msg->speed > 0.01f ? msg->speed : 1.0f, clip)) {
    RCLCPP_WARN(get_logger(), "unknown gesture id '%s'", msg->id.c_str());
    return;
  }

  clip_ = std::move(clip);
  clip_from_ = PostureKey{stand_height_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  clip_elapsed_ = 0.0;
  clip_loop_ = msg->loop;
  clip_id_ = msg->id;
  mode_ = MotionMode::Gesture;

  RCLCPP_INFO(
    get_logger(), "playing gesture '%s' (%zu keys, speed %.2f%s)",
    msg->id.c_str(), clip_.size(), static_cast<double>(msg->speed),
    msg->loop ? ", looping" : "");
}

// ---------------------------------------------------------------------------
// solvers
// ---------------------------------------------------------------------------

bool MotionNode::solve_posture(const PostureKey & pose, std::array<double, kNumJoints> & q)
{
  const Eigen::Matrix4d body_T = posture_transform(
    pose.height, pose.x, pose.y, pose.roll, pose.pitch, pose.yaw);
  return posture_ik(body_T, footprint_, link_params_, knee_, q);
}

bool MotionNode::solve_walk(double phase, std::array<double, kNumJoints> & q)
{
  // Trot: diagonal pairs swing together, so FL/RR share a phase and FR/RL are
  // half a cycle behind. Two feet are always down, which is the only reason a
  // static-margin-free gait works on a robot this small.
  const std::array<double, kNumLegs> offset{0.0, 0.5, 0.5, 0.0};   // FL FR RL RR

  const auto stance = nominal_stance(link_params_, stand_height_, splay_rad_);

  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const double leg_phase = std::fmod(phase + offset[i], 1.0);
    const Eigen::Vector3d & center = stance[i];

    // Foot velocity in the body frame is v + omega x r. Doing it per leg is what
    // makes turning work: the outer feet take a longer stride than the inner ones.
    const double vx = cmd_vel_.linear.x - cmd_vel_.angular.z * center(1);
    const double vy = cmd_vel_.linear.y + cmd_vel_.angular.z * center(0);
    const Eigen::Vector3d stride(vx * cycle_time_, vy * cycle_time_, 0.0);

    Eigen::Vector3d foot = center;
    if (leg_phase < duty_) {
      // Stance: the foot is planted, so it tracks backward under the body.
      const double u = leg_phase / duty_;
      foot -= stride * (u - 0.5);
    } else {
      // Swing: return to the front of the stroke, lifting on a half sine.
      const double u = (leg_phase - duty_) / (1.0 - duty_);
      foot += stride * (u - 0.5);
      foot(2) += step_height_ * std::sin(M_PI * u);
    }

    Eigen::Vector3d q_leg;
    if (!leg_ik(static_cast<Leg>(i), foot, link_params_, knee_, q_leg)) {
      return false;
    }
    q[3 * i + 0] = q_leg(0);
    q[3 * i + 1] = q_leg(1);
    q[3 * i + 2] = q_leg(2);
  }
  return true;
}

// ---------------------------------------------------------------------------
// main loop
// ---------------------------------------------------------------------------

void MotionNode::on_tick()
{
  const double dt = 1.0 / publish_rate_hz_;
  clock_ += dt;

  // Deadman: if teleop stops publishing, stop walking rather than trotting off
  // the edge of the desk.
  if (mode_ == MotionMode::Walk) {
    const double age = (now() - last_cmd_vel_stamp_).seconds();
    if (age > 0.5) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "cmd_vel stale (%.1f s) -- halting", age);
      mode_ = MotionMode::Idle;
      gait_phase_ = 0.0;
    }
  }

  std::array<double, kNumJoints> q{};
  bool solved = false;

  switch (mode_) {
    case MotionMode::Idle: {
      PostureKey rest{stand_height_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      rest.height += breathe_amp_ * std::sin(2.0 * M_PI * clock_ / breathe_period_);
      solved = solve_posture(rest, q);
      break;
    }

    case MotionMode::Posture:
      solved = solve_posture(posture_target_, q);
      break;

    case MotionMode::Walk:
      gait_phase_ = std::fmod(gait_phase_ + dt / cycle_time_, 1.0);
      solved = solve_walk(gait_phase_, q);
      break;

    case MotionMode::Gesture: {
      if (clip_.empty()) {
        mode_ = MotionMode::Idle;
        return;
      }
      const PostureKey & target = clip_.front();
      clip_elapsed_ += dt;

      const double u = target.duration > 1e-6 ? clip_elapsed_ / target.duration : 1.0;
      solved = solve_posture(blend(clip_from_, target, u), q);

      if (u >= 1.0) {
        clip_from_ = target;
        clip_.pop_front();
        clip_elapsed_ = 0.0;
        if (clip_.empty()) {
          if (clip_loop_) {
            load_clip(clip_id_, 1.0, clip_);
          } else {
            RCLCPP_INFO(get_logger(), "gesture '%s' done", clip_id_.c_str());
            mode_ = MotionMode::Idle;
          }
        }
      }
      break;
    }
  }

  if (!solved) {
    // Unreachable target. Hold the last good pose rather than publishing NaN --
    // NaN reaches the servos as a slam into the end stop.
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "IK unreachable in mode %d -- holding last pose", static_cast<int>(mode_));
    if (!have_last_q_) {
      return;
    }
    q = last_q_;
  }

  if (!clamp_to_limits(q, limits_)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "solution clamped to joint limits");
  }

  last_q_ = q;
  have_last_q_ = true;
  publish(q, dt);
}

void MotionNode::publish(const std::array<double, kNumJoints> & q, double duration)
{
  pkun_msgs::msg::JointCommand msg;
  msg.header.stamp = now();
  msg.name.assign(kJointNames.begin(), kJointNames.end());
  msg.position.assign(q.begin(), q.end());
  msg.duration = duration;
  command_pub_->publish(msg);
}

// ---------------------------------------------------------------------------
// clip table
// ---------------------------------------------------------------------------

bool MotionNode::load_clip(
  const std::string & id, double speed, std::deque<PostureKey> & clip) const
{
  const double rest = stand_height_;
  const double scale = 1.0 / std::max(0.05, speed);
  auto key = [&](double h, double x, double pitch_deg, double roll_deg,
      double yaw_deg, double seconds) {
      clip.push_back(
        PostureKey{h, x, 0.0, roll_deg * kDegToRad, pitch_deg * kDegToRad,
          yaw_deg * kDegToRad, seconds * scale});
    };

  clip.clear();

  // Body-level sketches of the clips authored in gestures.py. The head, ear,
  // eye and LED tracks of those same gestures live in pkun_expression -- this
  // node only owns the 12 leg DOF.
  if (id == "greet") {
    key(rest, 0.0, 8.0, 0.0, 0.0, 0.35);
    key(rest - 8.0, 6.0, -6.0, 0.0, 0.0, 0.30);
    key(rest, 0.0, 6.0, 0.0, 0.0, 0.30);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.35);
  } else if (id == "celebrate") {
    key(rest + 6.0, 0.0, 10.0, 0.0, 0.0, 0.25);
    key(rest - 6.0, 0.0, -4.0, 8.0, 0.0, 0.20);
    key(rest + 6.0, 0.0, 10.0, -8.0, 0.0, 0.20);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.30);
  } else if (id == "thinking") {
    key(rest, -4.0, -3.0, 6.0, 10.0, 0.50);
    key(rest, -4.0, -3.0, -6.0, -10.0, 0.60);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.45);
  } else if (id == "listening") {
    key(rest + 3.0, 3.0, 4.0, 0.0, 0.0, 0.40);
  } else if (id == "confirm_done") {
    key(rest, 0.0, -7.0, 0.0, 0.0, 0.18);
    key(rest, 0.0, 5.0, 0.0, 0.0, 0.18);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.25);
  } else if (id == "didnt_get_it") {
    key(rest, 0.0, 0.0, 9.0, 0.0, 0.30);
    key(rest, 0.0, 0.0, -9.0, 0.0, 0.30);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.30);
  } else if (id == "notice_low" || id == "notice_soon") {
    key(rest + 4.0, 2.0, 5.0, 0.0, 0.0, 0.30);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.35);
  } else if (id == "notice_urgent") {
    key(rest + 8.0, 4.0, 12.0, 0.0, 0.0, 0.15);
    key(rest - 4.0, -2.0, -6.0, 0.0, 0.0, 0.15);
    key(rest + 8.0, 4.0, 12.0, 0.0, 0.0, 0.15);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.25);
  } else if (id == "settle" || id == "home_reset") {
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.60);
  } else if (id == "focus_with_you") {
    key(rest + 2.0, 4.0, 6.0, 0.0, 18.0, 0.50);
  } else if (id == "comfort") {
    key(rest - 10.0, 2.0, -4.0, 0.0, 0.0, 0.70);
    key(rest - 6.0, 0.0, 0.0, 0.0, 0.0, 0.60);
  } else if (id == "speaking" || id == "idle_breathe") {
    key(rest + 2.0, 0.0, 1.0, 0.0, 0.0, 0.90);
    key(rest - 2.0, 0.0, -1.0, 0.0, 0.0, 0.90);
  } else if (id == "doze_off" || id == "low_power") {
    key(rest - 14.0, 0.0, -4.0, 0.0, 0.0, 1.20);
  } else if (id == "sleep_mode") {
    // Lowest the limits allow. find_sleep_pose() in sleep_wake.py searches for
    // the true minimum; this is the safe conservative version of it.
    key(rest - 20.0, 0.0, -3.0, 0.0, 0.0, 1.00);
    key(rest - 32.0, 0.0, 0.0, 0.0, 0.0, 1.20);
  } else if (id == "wake_up") {
    key(rest - 24.0, 0.0, 0.0, 0.0, 0.0, 0.60);
    key(rest + 4.0, 0.0, 8.0, 0.0, 0.0, 0.50);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.40);
  } else if (id == "error_state") {
    key(rest, 0.0, 0.0, 5.0, 0.0, 0.12);
    key(rest, 0.0, 0.0, -5.0, 0.0, 0.12);
    key(rest, 0.0, 0.0, 5.0, 0.0, 0.12);
    key(rest, 0.0, 0.0, 0.0, 0.0, 0.20);
  } else {
    return false;
  }

  return true;
}

}  // namespace pkun_motion

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pkun_motion::MotionNode)
