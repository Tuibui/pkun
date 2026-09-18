cmake_minimum_required(VERSION 3.8)
project(pkun_expression)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(pkun_msgs REQUIRED)

add_library(expression_component SHARED
  src/display.cpp
  src/sound_player.cpp
  src/expression_node.cpp
)
target_include_directories(expression_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(expression_component rclcpp rclcpp_components pkun_msgs)
rclcpp_components_register_nodes(expression_component "pkun_expression::ExpressionNode")

add_executable(expression_node src/expression_main.cpp)
target_link_libraries(expression_node expression_component)
ament_target_dependencies(expression_node rclcpp)

install(TARGETS expression_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS expression_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_display test/test_display.cpp src/display.cpp)
  target_include_directories(test_display PRIVATE include)
endif()

ament_export_include_directories(include)
ament_export_libraries(expression_component)
ament_export_dependencies(rclcpp pkun_msgs)
ament_package()
