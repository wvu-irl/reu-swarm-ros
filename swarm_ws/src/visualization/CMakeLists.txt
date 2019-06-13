cmake_minimum_required(VERSION 2.8.3)
project(visualization)

## Compile as C++11, supported in ROS Kinetic and newer
set (CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  wvu_swarm_std_msgs
  std_msgs
)

catkin_package(
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

#target_link_libraries(SFML sfml-graphics sfml-window sfml-system)
add_library(color_map src/color_map.cpp)
add_library(contour src/contour.cpp)

add_executable(table_vis src/table_vis.cpp)

target_link_libraries(color_map ${catkin_LIBRARIES})
target_link_libraries(contour ${catkin_LIBRARIES})
target_link_libraries(table_vis ${catkin_LIBRARIES} contour color_map)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
#Find any version 2.X of SFML
#See the FindSFML.cmake file for additional details and instructions
find_package(SFML 2 REQUIRED  graphics window system)
find_package(SFML 2 COMPONENTS graphics window system)
if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(table_vis ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
endif()

add_dependencies(table_vis ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) 

