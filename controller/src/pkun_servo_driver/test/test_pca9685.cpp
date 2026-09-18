#include <gtest/gtest.h>

#include <string>

#include "pkun_servo_driver/pca9685.hpp"

using pkun_servo_driver::Pca9685;

// These tests cover the parts that do not need hardware: argument validation and
// the prescale maths. Anything that actually talks to the bus is exercised on
// the robot, not here.

TEST(Pca9685, RejectsWritesWhenClosed)
{
  Pca9685 board("/dev/i2c-1", 0x40);
  EXPECT_FALSE(board.is_open());

  std::string error;
  EXPECT_FALSE(board.set_pwm(0, 0, 2048, error));
  EXPECT_FALSE(error.empty());
}

TEST(Pca9685, RejectsOutOfRangeChannel)
{
  Pca9685 board("/dev/i2c-1", 0x40);
  std::string error;
  EXPECT_FALSE(board.set_pwm(Pca9685::kNumChannels, 0, 100, error));
  EXPECT_NE(error.find("channel"), std::string::npos);
}

TEST(Pca9685, RejectsFrequencyOutsideChipRange)
{
  Pca9685 board("/dev/i2c-1", 0x40);
  std::string error;
  EXPECT_FALSE(board.set_frequency(5.0, error));
  EXPECT_FALSE(board.set_frequency(5000.0, error));
}

TEST(Pca9685, DefaultFrequencyMatchesAnalogServos)
{
  Pca9685 board("/dev/i2c-1", 0x40);
  EXPECT_DOUBLE_EQ(board.frequency(), 50.0);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
