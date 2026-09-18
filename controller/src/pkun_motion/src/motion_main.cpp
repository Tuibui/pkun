#include <memory>

#include "pkun_motion/motion_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pkun_motion::MotionNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
