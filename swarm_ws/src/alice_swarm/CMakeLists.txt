cmake_minimum_required(VERSION 2.8.3)
project(alice_swarm)

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
add_library(Hub src/Hub.cpp)
add_library(VectorQueue src/VectorQueue.cpp)
add_library(Rules src/Rules.cpp)
add_library(Robot src/Robot.cpp)
add_library(Model src/Model.cpp)

add_executable(AliceMain src/AliceMain.cpp)
target_link_libraries(Hub ${catkin_LIBRARIES})
target_link_libraries(VectorQueue ${catkin_LIBRARIES})
target_link_libraries(Rules ${catkin_LIBRARIES})
target_link_libraries(Robot ${catkin_LIBRARIES})
target_link_libraries(Model ${catkin_LIBRARIES})
target_link_libraries(AliceMain ${catkin_LIBRARIES})
  add_dependencies(AliceMain ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
