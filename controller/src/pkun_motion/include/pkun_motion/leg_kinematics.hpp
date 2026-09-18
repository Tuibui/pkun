#ifndef PKUN_MOTION__LEG_KINEMATICS_HPP_
#define PKUN_MOTION__LEG_KINEMATICS_HPP_

#include <array>
#include <string>

#include <Eigen/Dense>

namespace pkun_motion
{

/// Body frame {B}: x forward, y left, z up. All lengths in mm, angles in rad.
///
/// This is a direct port of leg_kinematics.py / walking_gait.py. The Python
/// versions are the reference implementation and have a verify() suite behind
/// them; if the two ever disagree, the Python is right and this is the bug.

enum class Leg : std::size_t {FL = 0, FR = 1, RL = 2, RR = 3};

inline constexpr std::size_t kNumLegs = 4;
inline constexpr std::size_t kJointsPerLeg = 3;
inline constexpr std::size_t kNumJoints = kNumLegs * kJointsPerLeg;

inline constexpr std::array<const char *, kNumLegs> kLegNames{"FL", "FR", "RL", "RR"};

/// Joint names as published on JointCommand, in the same order as kLegNames.
inline constexpr std::array<const char *, kNumJoints> kJointNames{
  "fl_abduction", "fl_hip_pitch", "fl_knee",
  "fr_abduction", "fr_hip_pitch", "fr_knee",
  "rl_abduction", "rl_hip_pitch", "rl_knee",
  "rr_abduction", "rr_hip_pitch", "rr_knee"};

/// Link dimensions, mm. Defaults are the measured P-kun values from Params in
/// leg_kinematics.py.
struct LinkParams
{
  double a{77.27};    // half hip spacing, front-back
  double b{29.0};     // half hip spacing, left-right
  double c{0.0};      // body centre down to the abduction axis
  double h{23.38};    // abduction axis down to the hip pitch axis
  double d1{12.75};   // lateral offset, hip pitch axis to the actual servo
  double d2{12.75};   // knee offset back toward the body
  double l1{56.39};   // thigh
  double l2{56.58};   // shank
};

/// Reachable range per joint, rad. Mirrors JointLimits in sleep_wake.py.
struct JointLimits
{
  double abduction_min{-55.0 * M_PI / 180.0};
  double abduction_max{55.0 * M_PI / 180.0};
  double hip_pitch_min{-90.0 * M_PI / 180.0};
  double hip_pitch_max{90.0 * M_PI / 180.0};
  double knee_min{0.0};                        // knee folds one way only
  double knee_max{150.0 * M_PI / 180.0};
};

/// Sign quad (a, b, D1, D2) for one leg.
std::array<double, 4> leg_signs(Leg leg);

/// Standard DH (Spong): Rz(theta) Tz(d) Tx(a) Rx(alpha).
Eigen::Matrix4d dh_transform(double theta, double d, double a, double alpha);

/// Fixed body -> frame 0 transform: Trans(+-a, +-b, -c) * RotY(90 deg).
Eigen::Matrix4d body_to_frame0(Leg leg, const LinkParams & p);

/// Hip position (origin of frame 0) in the body frame.
Eigen::Vector3d hip_position(Leg leg, const LinkParams & p);

/// Forward kinematics for one leg. Returns the foot position in the body frame.
Eigen::Vector3d leg_fk(Leg leg, const Eigen::Vector3d & q, const LinkParams & p);

/// Analytic inverse kinematics for one leg.
///
/// `knee` picks between the two 2-link solutions (+1 matches KNEE in gestures.py).
/// Returns false and leaves `q` untouched when the target is out of reach --
/// callers must check, because feeding NaN onward reaches the servos as a
/// full-speed slam to the joint limit.
bool leg_ik(
  Leg leg, const Eigen::Vector3d & foot, const LinkParams & p,
  int knee, Eigen::Vector3d & q);

/// 6-DOF body posture -> body-to-ground transform.
/// height/x/y in mm, roll/pitch/yaw in rad. + pitch = nose up.
Eigen::Matrix4d posture_transform(
  double height, double x, double y, double roll, double pitch, double yaw);

/// Neutral standing foot positions in the body frame, splayed outward.
std::array<Eigen::Vector3d, kNumLegs> nominal_stance(
  const LinkParams & p, double stand_height, double splay_rad, double x_center = 0.0);

/// Resting footprint on the ground plane (z = 0), used as the anchor for all
/// posture moves: the feet stay planted and the body moves over them.
std::array<Eigen::Vector3d, kNumLegs> footprint(
  const LinkParams & p, double stand_height, double splay_rad);

/// IK for all four legs with the body at `body_T` and the feet planted at
/// `feet_world`. Returns false if any leg cannot reach.
bool posture_ik(
  const Eigen::Matrix4d & body_T,
  const std::array<Eigen::Vector3d, kNumLegs> & feet_world,
  const LinkParams & p, int knee,
  std::array<double, kNumJoints> & q);

/// Clamp all 12 joint angles into their limits. Returns true if nothing needed
/// clamping.
bool clamp_to_limits(std::array<double, kNumJoints> & q, const JointLimits & limits);

}  // namespace pkun_motion

#endif  // PKUN_MOTION__LEG_KINEMATICS_HPP_
