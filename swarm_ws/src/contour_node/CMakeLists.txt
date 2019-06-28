cmake_minimum_required(VERSION 2.8.3)
project(contour_node)

# Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  genmsg
  roscpp
  rospy
  std_msgs
  wvu_swarm_std_msgs
  geometry_msgs
  visualization_msgs
  message_generation
)

add_message_files(
   FILES
   ellipse.msg
   gaussian.msg
   map_level.msg
   map_levels.msg
   obstacle.msg
)

generate_messages(DEPENDENCIES contour_node std_msgs wvu_swarm_std_msgs visualization_msgs geometry_msgs)

catkin_package(
   CATKIN_DEPENDS message_runtime
   INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(level_descriptor src/level_description.cpp)
add_dependencies(level_descriptor ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(level_descriptor ${catkin_LIBRARIES})

add_executable(mapping src/mapping.cpp)
add_dependencies(mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(mapping ${catkin_LIBRARIES})
 
