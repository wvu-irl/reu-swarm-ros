cmake_minimum_required(VERSION 2.8.3)
project(nuitrack_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -pthread)
add_definitions(-DCMAKE_BUILD_TYPE=Debug)
#add_definitions(-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0)

set(NUITRACK_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/../../nuitrack_sdk)
#message( ${NUITRACK_SDK_PATH} )

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)#
# is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_msgs
  visualization_msgs
  wvu_swarm_std_msgs
)
find_package(Boost REQUIRED COMPONENTS thread system regex)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

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
#  LIBRARIES nuitrack_ros
   CATKIN_DEPENDS roscpp geometry_msgs std_msgs visualization_msgs wvu_swarm_std_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${NUITRACK_SDK_PATH}/Nuitrack/include   
  ${NUITRACK_SDK_PATH}/Nuitrack/include/middleware  
)

add_executable(hand_pub src/hand_pub.cpp)
add_dependencies(hand_pub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})  
target_link_libraries(
  hand_pub
  ${catkin_LIBRARIES}
  ${NUITRACK_SDK_PATH}/Nuitrack/lib/linux64/libnuitrack.so
  ${NUITRACK_SDK_PATH}/Nuitrack/lib/linux64/libmiddleware.so
  ${Boost_LIBRARIES}
)