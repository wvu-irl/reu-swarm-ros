cmake_minimum_required(VERSION 2.8.3)
project(swarm_simulation)

set (CMAKE_CXX_STANDARD 11)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs wvu_swarm_std_msgs geometry_msgs swarm_server)

## Declare ROS messages and services
##add_message_files(FILES )

## Generate added messages and services

## Declare a catkin package contour_node geometry_msgs visualization_msgs)

catkin_package(
	INCLUDE_DIRS include
)

## Build talker and listener

# Detect and add SFML

include_directories(include ${catkin_INCLUDE_DIRS})
add_library(Body src/Body.cpp)
add_library(Flock src/Flock.cpp)
add_library(Sim src/Sim.cpp)
add_library(Pvector src/Pvector.cpp)
add_library(Hawk_Sim src/hawk_sim.cpp)

add_executable(main src/main.cpp)
add_executable(hawk_sim_execute src/hawk_sim_execute.cpp)   ##runs hawk sim stuff

add_dependencies(Body wvu_swarm_std_msgs_generate_messages_cpp)
add_dependencies(Flock wvu_swarm_std_msgs_generate_messages_cpp)
add_dependencies(Hawk_Sim ${catkin_LIBRARIES} wvu_swarm_std_msgs_generate_messages_cpp)
add_dependencies(hawk_sim_execute ${catkin_LIBRARIES} wvu_swarm_std_msgs_generate_messages_cpp)

target_link_libraries(main ${catkin_LIBRARIES} Sim)
target_link_libraries(Sim ${catkin_LIBRARIES} Flock Body Pvector)
target_link_libraries(Body ${catkin_LIBRARIES}) 
target_link_libraries(Flock ${catkin_LIBRARIES})

target_link_libraries(hawk_sim_execute ${catkin_LIBRARIES} Hawk_Sim) ##runs hawk sim stuff
	

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
#Find any version 2.X of SFML
#See the FindSFML.cmake file for additional details and instructions
find_package(SFML 2 REQUIRED  graphics window system)
find_package(SFML 2 COMPONENTS graphics window system)
if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(main ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
endif()

add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
