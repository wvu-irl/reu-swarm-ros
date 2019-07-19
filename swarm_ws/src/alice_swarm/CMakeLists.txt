cmake_minimum_required(VERSION 2.8.3)
project(alice_swarm)

set (CMAKE_CXX_STANDARD 11)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg message_generation wvu_swarm_std_msgs swarm_server  contour_node geometry_msgs visualization_msgs)

## Declare ROS messages and services
##add_message_files(FILES )

add_service_files(
  FILES
  get_map.srv
  get_maps.srv
)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs wvu_swarm_std_msgs)

## Declare a catkin package
catkin_package(
)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})
add_library(Hub src/Hub.cpp)
add_library(Rules src/Rules.cpp)
add_library(Alice src/Alice.cpp)
add_library(Model src/Model.cpp)

add_dependencies(Hub wvu_swarm_std_msgs_generate_messages_cpp)
add_dependencies(Model alice_swarm_generate_messages_cpp)
add_dependencies(Rules ${catkin_LIBRARIES} wvu_swarm_std_msgs_generate_messages_cpp)
add_executable(AliceMain src/AliceMain.cpp)
add_executable(AbsToRel src/AbsToRel.cpp)
add_executable(InterHub src/InterHub.cpp)
target_link_libraries(Hub ${catkin_LIBRARIES} level_descriptor)
target_link_libraries(Rules ${catkin_LIBRARIES})
target_link_libraries(Alice ${catkin_LIBRARIES} Model Rules)
target_link_libraries(Model ${catkin_LIBRARIES})
target_link_libraries(InterHub ${catkin_LIBRARIES})
 add_dependencies(InterHub alice_swarm_generate_messages_cpp)
target_link_libraries(AbsToRel ${catkin_LIBRARIES} Hub)
target_link_libraries(AliceMain ${catkin_LIBRARIES} Alice Hub)
  add_dependencies(AliceMain ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
