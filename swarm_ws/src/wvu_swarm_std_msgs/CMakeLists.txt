cmake_minimum_required(VERSION 2.8.3)
project(wvu_swarm_std_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
  tf2
  tf2_ros
  visualization_msgs
  genmsg
)

## Generate messages in the 'msg' folder
  add_message_files(
    FILES
    aliceMail.msg
    aliceMailArray.msg
    rtheta.msg
    viconBot.msg
    viconBotArray.msg
    typedPoint.msg
    robotcommand.msg
    sensor_data.msg
  )

## Generate added messages and services with any dependencies listed here
  generate_messages(
    DEPENDENCIES
    geometry_msgs
    std_msgs
    visualization_msgs
    wvu_swarm_std_msgs
  )

catkin_package(
   CATKIN_DEPENDS message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
