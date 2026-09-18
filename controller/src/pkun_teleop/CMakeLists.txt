cmake_minimum_required(VERSION 3.8)
project(pkun_teleop)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pkun_msgs REQUIRED)

add_library(joy_teleop_component SHARED src/joy_teleop_node.cpp)
target_include_directories(joy_teleop_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(joy_teleop_component
  rclcpp rclcpp_components sensor_msgs geometry_msgs pkun_msgs)
rclcpp_components_register_nodes(joy_teleop_component "pkun_teleop::JoyTeleopNode")

add_executable(joy_teleop_node src/joy_teleop_main.cpp)
target_link_libraries(joy_teleop_node joy_teleop_component)
ament_target_dependencies(joy_teleop_node rclcpp)

install(TARGETS joy_teleop_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS joy_teleop_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(joy_teleop_component)
ament_export_dependencies(rclcpp sensor_msgs geometry_msgs pkun_msgs)
ament_package()
