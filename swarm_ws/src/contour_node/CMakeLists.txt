cmake_minimum_required(VERSION 2.8.3)
project(contour_node)

# Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nuitrack_bridge
  wvu_swarm_std_msgs
)

catkin_package(
   INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(level_object src/level_object.cpp)
add_dependencies(level_object wvu_swarm_std_msgs_generate_messages_cpp)
target_link_libraries(level_object ${catkin_LIBRARIES})

add_library(ellipse_object src/ellipse_object.cpp)
add_dependencies(ellipse_object wvu_swarm_std_msgs_generate_messages_cpp)
target_link_libraries(ellipse_object ${catkin_LIBRARIES} level_object)

add_library(gaussian_object src/gaussian_object.cpp)
add_dependencies(gaussian_object wvu_swarm_std_msgs_generate_messages_cpp)
target_link_libraries(gaussian_object ${catkin_LIBRARIES} ellipse_object)

add_library(level_descriptor src/level_description.cpp)
add_dependencies(level_descriptor wvu_swarm_std_msgs_generate_messages_cpp)
target_link_libraries(level_descriptor ${catkin_LIBRARIES} gaussian_object)

add_library(universe src/universe_object.cpp)
add_dependencies(universe wvu_swarm_std_msgs_generate_messages_cpp)
target_link_libraries(universe ${catkin_LIBRARIES} gaussian_object)

add_executable(mapping src/mapping.cpp)
add_dependencies(mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} wvu_swarm_std_msgs_generate_messages_cpp universe)
target_link_libraries(mapping ${catkin_LIBRARIES} gaussian_object universe)