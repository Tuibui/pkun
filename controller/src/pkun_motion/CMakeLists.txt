cmake_minimum_required(VERSION 3.8)
project(pkun_motion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pkun_msgs REQUIRED)

# Kinematics is a plain library with no ROS in it, so the tests can link it
# without spinning up a node.
add_library(pkun_kinematics SHARED src/leg_kinematics.cpp)
target_include_directories(pkun_kinematics PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(pkun_kinematics Eigen3::Eigen)

add_library(motion_component SHARED src/motion_node.cpp)
target_include_directories(motion_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(motion_component pkun_kinematics Eigen3::Eigen)
ament_target_dependencies(motion_component
  rclcpp rclcpp_components geometry_msgs pkun_msgs)
rclcpp_components_register_nodes(motion_component "pkun_motion::MotionNode")

add_executable(motion_node src/motion_main.cpp)
target_link_libraries(motion_node motion_component)
ament_target_dependencies(motion_node rclcpp)

install(TARGETS pkun_kinematics motion_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS motion_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_leg_kinematics test/test_leg_kinematics.cpp)
  target_link_libraries(test_leg_kinematics pkun_kinematics Eigen3::Eigen)
  target_include_directories(test_leg_kinematics PRIVATE include)
endif()

ament_export_include_directories(include)
ament_export_libraries(pkun_kinematics motion_component)
ament_export_dependencies(rclcpp geometry_msgs pkun_msgs Eigen3)
ament_package()
