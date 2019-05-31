cmake_minimum_required(VERSION 2.8.3)
project(processor)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg message_generation wvu_swarm_std_msgs swarm_server)

## Declare ROS messages and services
##add_message_files(FILES )

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

add_library(Processor src/Processor.cpp)
add_executable(ProRun src/ProRun.cpp)
target_link_libraries(ProRun ${catkin_LIBRARIES})
  add_dependencies(ProRun ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
