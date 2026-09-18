#include <memory>

#include "pkun_servo_driver/servo_driver_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Single-threaded on purpose: every callback here ends up writing to the same
  // I2C file descriptor, so there is nothing to gain from parallelism and a real
  // risk of interleaving two register writes.
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<pkun_servo_driver::ServoDriverNode>(rclcpp::NodeOptions());
  executor.add_node(node->get_node_base_interface());
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
