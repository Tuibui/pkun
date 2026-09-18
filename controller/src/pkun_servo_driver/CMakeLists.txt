cmake_minimum_required(VERSION 3.8)
project(pkun_servo_driver)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(pkun_msgs REQUIRED)

add_library(servo_driver_component SHARED
  src/pca9685.cpp
  src/servo_driver_node.cpp
)
target_include_directories(servo_driver_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(servo_driver_component
  rclcpp rclcpp_components rclcpp_lifecycle lifecycle_msgs pkun_msgs)
rclcpp_components_register_nodes(servo_driver_component
  "pkun_servo_driver::ServoDriverNode")

add_executable(servo_driver_node src/servo_driver_main.cpp)
target_link_libraries(servo_driver_node servo_driver_component)
ament_target_dependencies(servo_driver_node rclcpp)

install(TARGETS servo_driver_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS servo_driver_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_pca9685 test/test_pca9685.cpp src/pca9685.cpp)
  target_include_directories(test_pca9685 PRIVATE include)
endif()

ament_export_include_directories(include)
ament_export_libraries(servo_driver_component)
ament_export_dependencies(rclcpp rclcpp_lifecycle pkun_msgs)
ament_package()
