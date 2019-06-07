cmake_minimum_required(VERSION 2.8.3)
project(processor)

set (CMAKE_CXX_STANDARD 11)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg message_generation wvu_swarm_std_msgs swarm_server geometry_msgs visualization_msgs)

## Declare ROS messages and services
##add_message_files(FILES )

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs wvu_swarm_std_msgs)

## Declare a catkin package
catkin_package(
)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

add_library(Body src/Body.cpp)
add_library(Flock src/Flock.cpp)
add_library(Pvector src/Pvector.cpp)

add_executable(main src/main.cpp)

target_link_libraries(Processor ${catkin_LIBRARIES})
target_link_libraries(ProRun ${catkin_LIBRARIES})
  add_dependencies(ProRun ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

 
