cmake_minimum_required(VERSION 2.8.3)
project(visualization)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
#find_package(sfml COMPONENTS graphics window system REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  wvu_swarm_std_msgs
  std_msgs
  geometry_msgs
  contour_node
  swarm_simulation
  alice_swarm
  swarm_simulation
)

catkin_package(
	INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

find_package(SFML)

if (SFML_FOUND)
	find_package(SFML 2 COMPONENTS graphics window system)

	ADD_LIBRARY(color_map ${LIB_TYPE} src/color_map.cpp)
	target_link_libraries(color_map ${catkin_LIBRARIES})
	
	ADD_LIBRARY(alice_pov ${LIB_TYPE} src/alice_pov.cpp)
	target_link_libraries(alice_pov ${catkin_LIBRARIES})
	add_dependencies(alice_pov ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} wvu_swarm_std_msgs_generate_messages_cpp)
	
	add_executable(alice_pov_main src/alice_pov_main.cpp)
	target_link_libraries(alice_pov_main ${SFML_LIBRARIES} ${SFML_DEPENDENCIES} ${catkin_LIBRARIES} alice_pov)
	add_dependencies(alice_pov_main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} wvu_swarm_std_msgs_generate_messages_cpp)
	
	add_executable(calibrate_tab src/calibrate.cpp)
	target_link_libraries(calibrate_tab ${SFML_LIBRARIES} ${SFML_DEPENDENCIES} ${catkin_LIBRARIES})
	add_dependencies(calibrate_tab ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
	
	find_package(CUDA)
	if (CUDA_FOUND)

		list(APPEND sms 30 35 37 50 52 60 61 70)
		foreach(sm IN LISTS sms)
			set(ARCH_SMS "${ARCH_SMS} -gencode arch=compute_${sm},code=sm_${sm}")
		endforeach(sm)
	
		set(CUDA_NVCC_FLAGS "${ARCH_SMS} -ccbin g++" CACHE STRING "nvcc flags" FORCE)
		set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc" FORCE)
		set(LIB_TYPE STATIC)
		
		CUDA_ADD_LIBRARY(contour ${LIB_TYPE} src/contour.cu)
		add_dependencies(contour ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} wvu_swarm_std_msgs_generate_messages_cpp)
		target_link_libraries(contour ${catkin_LIBRARIES} ${CUDA_LIBRARIES} color_map)
	
		add_library(keyboard src/keyboard_managment.cpp)
		add_dependencies(keyboard ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
		target_link_libraries(keyboard ${catkin_LIBRARIES})
		
		CUDA_ADD_LIBRARY(perspective_tf ${LIB_TYPE} src/transform/perspective_transform_gpu.cu)
		target_link_libraries(perspective_tf ${catkin_LIBRARIES} ${CUDA_LIBRARIES})
				
		add_executable(table_vis src/table_vis.cpp)
		add_dependencies(table_vis ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} wvu_swarm_std_msgs_generate_messages_cpp)
		target_link_libraries(table_vis ${SFML_LIBRARIES} ${SFML_DEPENDENCIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES} contour perspective_tf keyboard)
	
	endif()
else()
	message(ERROR_VARIABLE "Could not locate SFML")
endif()