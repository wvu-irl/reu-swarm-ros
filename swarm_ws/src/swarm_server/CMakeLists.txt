cmake_minimum_required(VERSION 2.8.3)
project(swarm_server)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  vicon_demo
  wvu_swarm_std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES swarm_server
#  CATKIN_DEPENDS roscpp rospy std_msgs vicon_demo
#  DEPENDS system_lib
)

add_library(robot_id src/robot_id.cpp)
target_link_libraries(robot_id ${catkin_LIBRARIES})

add_library(arduino_server src/arduino_server_source.cpp)

add_executable(ros_to_arduino_server src/ros_to_arduino_server.cpp)
target_link_libraries(ros_to_arduino_server ${catkin_LIBRARIES})
add_dependencies(ros_to_arduino_server ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})  

add_executable(funnel src/funnel.cpp)
target_link_libraries(funnel ${catkin_LIBRARIES})
add_dependencies(funnel ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) 

add_executable(testing src/testing.cpp)
target_link_libraries(testing ${catkin_LIBRARIES})
add_dependencies(testing ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})   
