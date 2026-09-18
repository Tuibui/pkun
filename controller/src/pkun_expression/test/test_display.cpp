#include <gtest/gtest.h>

#include <string>

#include "pkun_expression/display.hpp"

using namespace pkun_expression;  // NOLINT(build/namespaces)

TEST(Display, FactoryKnowsItsBackends)
{
  std::string error;
  EXPECT_NE(make_display("console", "/dev/i2c-1", 0x3C, error), nullptr);
  EXPECT_NE(make_display("ssd1306", "/dev/i2c-1", 0x3C, error), nullptr);

  error.clear();
  EXPECT_EQ(make_display("nonsense", "/dev/i2c-1", 0x3C, error), nullptr);
  EXPECT_FALSE(error.empty());
}

TEST(ConsoleDisplay, OpenEyeAndClosedEyeRenderDifferently)
{
  ConsoleDisplay display;
  std::string error;
  ASSERT_TRUE(display.open(error));

  FaceState face;
  face.eye = 1.0;
  ASSERT_TRUE(display.render(face, error));
  const std::string open_frame = display.last_frame();

  face.eye = 0.0;
  ASSERT_TRUE(display.render(face, error));
  EXPECT_NE(open_frame, display.last_frame());
}

TEST(ConsoleDisplay, ShowsLedNameAndBrightness)
{
  ConsoleDisplay display;
  std::string error;
  ASSERT_TRUE(display.open(error));

  FaceState face;
  face.led_name = "urgent";
  face.led_v = 1.0;
  ASSERT_TRUE(display.render(face, error));

  EXPECT_NE(display.last_frame().find("urgent"), std::string::npos);
  EXPECT_NE(display.last_frame().find("100%"), std::string::npos);
}

TEST(Ssd1306Display, RenderFailsCleanlyWhenNotOpen)
{
  Ssd1306Display display("/dev/i2c-does-not-exist", 0x3C);
  std::string error;

  EXPECT_FALSE(display.open(error));
  EXPECT_FALSE(error.empty());

  error.clear();
  FaceState face;
  EXPECT_FALSE(display.render(face, error));
  EXPECT_FALSE(error.empty());
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
