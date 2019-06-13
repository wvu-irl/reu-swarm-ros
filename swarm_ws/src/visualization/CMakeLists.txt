cmake_minimum_required(VERSION 2.8.3)
project(visualization)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lsfml-graphics -lsfml-window -lsfml-system")
#find_package(sfml COMPONENTS graphics window system REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  wvu_swarm_std_msgs
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES visualization
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS libsfml-dev
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#target_link_libraries(SFML sfml-graphics sfml-window sfml-system)

add_library(color_map src/color_map.cpp)
target_link_libraries(color_map ${catkin_LIBRARIES})

add_library(contour src/contour.cpp)
target_link_libraries(contour ${catkin_LIBRARIES})

