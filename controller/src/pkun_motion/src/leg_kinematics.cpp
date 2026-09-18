#include "pkun_motion/leg_kinematics.hpp"

#include <algorithm>
#include <cmath>

namespace pkun_motion
{

std::array<double, 4> leg_signs(Leg leg)
{
  // (sign_a, sign_b, sign_D1, sign_D2) -- must match LEG_SIGNS in leg_kinematics.py
  switch (leg) {
    case Leg::FL: return {+1.0, +1.0, +1.0, +1.0};
    case Leg::FR: return {+1.0, -1.0, -1.0, -1.0};
    case Leg::RL: return {-1.0, +1.0, +1.0, +1.0};
    case Leg::RR: return {-1.0, -1.0, -1.0, -1.0};
  }
  return {+1.0, +1.0, +1.0, +1.0};
}

Eigen::Matrix4d dh_transform(double theta, double d, double a, double alpha)
{
  const double ct = std::cos(theta), st = std::sin(theta);
  const double ca = std::cos(alpha), sa = std::sin(alpha);

  Eigen::Matrix4d T;
  T << ct, -st * ca,  st * sa, a * ct,
    st,  ct * ca, -ct * sa, a * st,
    0.0,       sa,       ca,      d,
    0.0,      0.0,      0.0,    1.0;
  return T;
}

Eigen::Matrix4d body_to_frame0(Leg leg, const LinkParams & p)
{
  const auto s = leg_signs(leg);

  // RotY(+90 deg)
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T(0, 0) = 0.0; T(0, 2) = 1.0;
  T(2, 0) = -1.0; T(2, 2) = 0.0;

  T(0, 3) = s[0] * p.a;
  T(1, 3) = s[1] * p.b;
  T(2, 3) = -p.c;
  return T;
}

Eigen::Vector3d hip_position(Leg leg, const LinkParams & p)
{
  const auto s = leg_signs(leg);
  return {s[0] * p.a, s[1] * p.b, -p.c};
}

Eigen::Vector3d leg_fk(Leg leg, const Eigen::Vector3d & q, const LinkParams & p)
{
  const auto s = leg_signs(leg);

  Eigen::Matrix4d T = body_to_frame0(leg, p);
  T *= dh_transform(q(0), 0.0, p.h, -M_PI / 2.0);
  T *= dh_transform(q(1), s[2] * p.d1, p.l1, 0.0);
  T *= dh_transform(q(2), -s[3] * p.d2, p.l2, 0.0);
  return T.block<3, 1>(0, 3);
}

bool leg_ik(
  Leg leg, const Eigen::Vector3d & foot, const LinkParams & p,
  int knee, Eigen::Vector3d & q)
{
  const auto s = leg_signs(leg);
  const Eigen::Vector3d r = foot - hip_position(leg, p);
  const double rx = r(0), ry = r(1), rz = r(2);

  // Net lateral offset, always along the z1 axis.
  const double d_net = s[2] * p.d1 - s[3] * p.d2;
  const double radius = std::hypot(ry, rz);
  if (radius * radius < d_net * d_net) {
    return false;   // too close to the abduction axis for the offset to clear
  }

  const double reach = std::sqrt(radius * radius - d_net * d_net);
  const double lambda = reach - p.h;               // what the 2-link chain must cover
  const double th1 = std::atan2(ry, -rz) - std::atan2(d_net, reach);

  // 2-link in the leg plane: lambda is "down the leg", m is along body x.
  const double m = -rx;                            // +theta2 pushes the foot toward -x
  const double rho2 = lambda * lambda + m * m;
  const double cos3 = (rho2 - p.l1 * p.l1 - p.l2 * p.l2) / (2.0 * p.l1 * p.l2);
  if (std::abs(cos3) > 1.0) {
    return false;   // out of reach, or folded tighter than the links allow
  }

  const double th3 = static_cast<double>(knee) * std::acos(cos3);
  const double th2 = std::atan2(m, lambda) -
    std::atan2(p.l2 * std::sin(th3), p.l1 + p.l2 * std::cos(th3));

  q = Eigen::Vector3d(th1, th2, th3);
  return true;
}

Eigen::Matrix4d posture_transform(
  double height, double x, double y, double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);

  Eigen::Matrix3d rx;
  rx << 1.0, 0.0, 0.0,
    0.0, cr, -sr,
    0.0, sr, cr;

  // Note the sign layout: this is the "+ pitch = nose up" convention used by
  // posture_T in gestures.py, not a textbook RotY.
  Eigen::Matrix3d ry;
  ry << cp, 0.0, -sp,
    0.0, 1.0, 0.0,
    sp, 0.0, cp;

  Eigen::Matrix3d rz;
  rz << cy, -sy, 0.0,
    sy, cy, 0.0,
    0.0, 0.0, 1.0;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = rz * ry * rx;
  T(0, 3) = x;
  T(1, 3) = y;
  T(2, 3) = height;
  return T;
}

std::array<Eigen::Vector3d, kNumLegs> nominal_stance(
  const LinkParams & p, double stand_height, double splay_rad, double x_center)
{
  std::array<Eigen::Vector3d, kNumLegs> stance;
  const double reach = stand_height - p.h;

  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const auto leg = static_cast<Leg>(i);
    const auto s = leg_signs(leg);
    const Eigen::Vector3d hip = hip_position(leg, p);

    // Splaying rotates about the abduction axis, pushing the foot outward.
    stance[i] = Eigen::Vector3d(
      hip(0) + x_center,
      hip(1) + s[1] * reach * std::tan(splay_rad),
      hip(2) - stand_height);
  }
  return stance;
}

std::array<Eigen::Vector3d, kNumLegs> footprint(
  const LinkParams & p, double stand_height, double splay_rad)
{
  auto stance = nominal_stance(p, stand_height, splay_rad);
  for (auto & foot : stance) {
    foot(2) = 0.0;    // drop onto the ground plane
  }
  return stance;
}

bool posture_ik(
  const Eigen::Matrix4d & body_T,
  const std::array<Eigen::Vector3d, kNumLegs> & feet_world,
  const LinkParams & p, int knee,
  std::array<double, kNumJoints> & q)
{
  const Eigen::Matrix3d r_transpose = body_T.block<3, 3>(0, 0).transpose();
  const Eigen::Vector3d origin = body_T.block<3, 1>(0, 3);

  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const Eigen::Vector3d target = r_transpose * (feet_world[i] - origin);

    Eigen::Vector3d q_leg;
    if (!leg_ik(static_cast<Leg>(i), target, p, knee, q_leg)) {
      return false;
    }
    q[3 * i + 0] = q_leg(0);
    q[3 * i + 1] = q_leg(1);
    q[3 * i + 2] = q_leg(2);
  }
  return true;
}

bool clamp_to_limits(std::array<double, kNumJoints> & q, const JointLimits & limits)
{
  const std::array<double, kJointsPerLeg> lo{
    limits.abduction_min, limits.hip_pitch_min, limits.knee_min};
  const std::array<double, kJointsPerLeg> hi{
    limits.abduction_max, limits.hip_pitch_max, limits.knee_max};

  bool within = true;
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    const std::size_t j = i % kJointsPerLeg;
    const double clamped = std::clamp(q[i], lo[j], hi[j]);
    if (std::abs(clamped - q[i]) > 1e-9) {
      within = false;
    }
    q[i] = clamped;
  }
  return within;
}

}  // namespace pkun_motion
