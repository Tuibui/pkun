#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "pkun_motion/leg_kinematics.hpp"

using namespace pkun_motion;  // NOLINT(build/namespaces)

namespace
{
constexpr double kTol = 1e-6;

LinkParams params()
{
  return LinkParams{};
}
}  // namespace

// Same checks verify() runs in leg_kinematics.py. If these fail, the DH table or
// a leg sign has been mistyped in the C++ port.

TEST(LegKinematics, ZeroPoseMatchesClosedForm)
{
  const auto p = params();
  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const auto leg = static_cast<Leg>(i);
    const auto s = leg_signs(leg);

    const Eigen::Vector3d expected(
      s[0] * p.a,
      s[1] * p.b + s[2] * p.d1 - s[3] * p.d2,
      -(p.c + p.h + p.l1 + p.l2));

    const Eigen::Vector3d got = leg_fk(leg, Eigen::Vector3d::Zero(), p);
    EXPECT_NEAR(got(0), expected(0), kTol) << "leg " << kLegNames[i];
    EXPECT_NEAR(got(1), expected(1), kTol) << "leg " << kLegNames[i];
    EXPECT_NEAR(got(2), expected(2), kTol) << "leg " << kLegNames[i];
  }
}

TEST(LegKinematics, AbductionSweepsACircleAboutZ0)
{
  const auto p = params();
  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const auto leg = static_cast<Leg>(i);
    const Eigen::Vector3d hip = hip_position(leg, p);

    double x_min = 1e9, x_max = -1e9, r_min = 1e9, r_max = -1e9;
    for (int k = -10; k <= 10; ++k) {
      const double th1 = 0.1 * k;
      const Eigen::Vector3d foot = leg_fk(leg, Eigen::Vector3d(th1, 0.0, 0.0), p);
      x_min = std::min(x_min, foot(0));
      x_max = std::max(x_max, foot(0));
      const double radius = std::hypot(foot(1) - hip(1), foot(2) - hip(2));
      r_min = std::min(r_min, radius);
      r_max = std::max(r_max, radius);
    }
    EXPECT_NEAR(x_max - x_min, 0.0, kTol) << "leg " << kLegNames[i] << " x drifted";
    EXPECT_NEAR(r_max - r_min, 0.0, kTol) << "leg " << kLegNames[i] << " radius drifted";
  }
}

TEST(LegKinematics, HipAndKneeDoNotMoveFootSideways)
{
  const auto p = params();
  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const auto leg = static_cast<Leg>(i);
    for (int joint : {1, 2}) {
      double y_min = 1e9, y_max = -1e9;
      for (int k = -10; k <= 10; ++k) {
        Eigen::Vector3d q = Eigen::Vector3d::Zero();
        q(joint) = 0.1 * k;
        const double y = leg_fk(leg, q, p)(1);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
      }
      EXPECT_NEAR(y_max - y_min, 0.0, kTol)
        << "leg " << kLegNames[i] << " joint " << joint;
    }
  }
}

TEST(LegKinematics, IkInvertsFk)
{
  const auto p = params();
  const std::array<Eigen::Vector3d, 3> poses{
    Eigen::Vector3d(0.20, -0.35, 0.80),
    Eigen::Vector3d(-0.15, 0.25, 1.10),
    Eigen::Vector3d(0.00, 0.00, 0.60)};

  for (std::size_t i = 0; i < kNumLegs; ++i) {
    const auto leg = static_cast<Leg>(i);
    for (const auto & q_ref : poses) {
      const Eigen::Vector3d foot = leg_fk(leg, q_ref, p);

      Eigen::Vector3d q_ik;
      ASSERT_TRUE(leg_ik(leg, foot, p, +1, q_ik))
        << "leg " << kLegNames[i] << " reported its own FK pose unreachable";

      // Recover the same foot, not necessarily the same angles -- the 2-link
      // chain has two solutions and `knee` picks the branch.
      const Eigen::Vector3d round_trip = leg_fk(leg, q_ik, p);
      EXPECT_NEAR(round_trip(0), foot(0), 1e-6) << "leg " << kLegNames[i];
      EXPECT_NEAR(round_trip(1), foot(1), 1e-6) << "leg " << kLegNames[i];
      EXPECT_NEAR(round_trip(2), foot(2), 1e-6) << "leg " << kLegNames[i];
    }
  }
}

TEST(LegKinematics, IkRejectsTargetsBeyondReach)
{
  const auto p = params();
  Eigen::Vector3d q;

  // Far beyond L1 + L2.
  EXPECT_FALSE(leg_ik(Leg::FL, Eigen::Vector3d(0.0, 0.0, -1000.0), p, +1, q));

  // The reachable annulus also has an inner edge at |L1 - L2|. On stock P-kun
  // the links are within 0.2 mm of each other, so the leg folds essentially
  // flat and that edge is unobservable -- shorten the shank to expose it.
  auto stubby = params();
  stubby.l2 = 20.0;                       // inner limit becomes 36.4 mm
  const Eigen::Vector3d hip = hip_position(Leg::FL, stubby);
  EXPECT_FALSE(leg_ik(Leg::FL, hip + Eigen::Vector3d(0.0, 0.0, -stubby.h), stubby, +1, q));
}

TEST(LegKinematics, IkRejectsTargetsInsideTheLateralOffset)
{
  // The "too close to the abduction axis" branch only exists when D1 != D2. On
  // stock P-kun they are equal, the net offset is zero, and a target on the axis
  // is perfectly reachable by folding the leg -- so build an asymmetric leg to
  // exercise it.
  auto p = params();
  p.d2 = 0.0;    // net lateral offset becomes D1 = 12.75 mm

  Eigen::Vector3d q;
  const Eigen::Vector3d hip = hip_position(Leg::FL, p);

  // Sitting exactly on the abduction axis: the offset link cannot fold to zero.
  EXPECT_FALSE(leg_ik(Leg::FL, hip, p, +1, q));

  // Just outside the offset radius it becomes reachable again.
  EXPECT_TRUE(leg_ik(Leg::FL, hip + Eigen::Vector3d(0.0, 0.0, -100.0), p, +1, q));
}

TEST(LegKinematics, NominalStanceStandsAtTheRequestedHeight)
{
  const auto p = params();
  const double stand_h = 110.0;
  const auto stance = nominal_stance(p, stand_h, 12.0 * M_PI / 180.0);

  for (std::size_t i = 0; i < kNumLegs; ++i) {
    EXPECT_NEAR(stance[i](2), -p.c - stand_h, kTol) << "leg " << kLegNames[i];

    Eigen::Vector3d q;
    EXPECT_TRUE(leg_ik(static_cast<Leg>(i), stance[i], p, +1, q))
      << "rest stance is unreachable for leg " << kLegNames[i];
  }
}

TEST(LegKinematics, RestPostureIsWithinJointLimits)
{
  const auto p = params();
  const auto feet = footprint(p, 110.0, 12.0 * M_PI / 180.0);
  const Eigen::Matrix4d body_T = posture_transform(110.0, 0, 0, 0, 0, 0);

  std::array<double, kNumJoints> q{};
  ASSERT_TRUE(posture_ik(body_T, feet, p, +1, q));

  auto before = q;
  EXPECT_TRUE(clamp_to_limits(q, JointLimits{}))
    << "the rest stance itself violates a joint limit";
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_NEAR(before[i], q[i], kTol) << kJointNames[i];
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
