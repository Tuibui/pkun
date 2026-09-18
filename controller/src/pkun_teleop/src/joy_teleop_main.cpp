#include <memory>

#include "pkun_teleop/joy_teleop_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pkun_teleop::JoyTeleopNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
