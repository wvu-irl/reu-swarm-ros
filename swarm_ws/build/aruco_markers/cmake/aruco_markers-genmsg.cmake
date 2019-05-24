# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "aruco_markers: 2 messages, 0 services")

set(MSG_I_FLAGS "-Iaruco_markers:/home/air/swarm_ws/src/aruco_markers/msg;-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/kinetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(aruco_markers_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" NAME_WE)
add_custom_target(_aruco_markers_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "aruco_markers" "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" "geometry_msgs/Pose2D:std_msgs/Header:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" NAME_WE)
add_custom_target(_aruco_markers_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "aruco_markers" "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" "aruco_markers/Marker:geometry_msgs/Pose2D:std_msgs/Header:geometry_msgs/Vector3"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/aruco_markers
)
_generate_msg_cpp(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg"
  "${MSG_I_FLAGS}"
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/aruco_markers
)

### Generating Services

### Generating Module File
_generate_module_cpp(aruco_markers
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/aruco_markers
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(aruco_markers_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(aruco_markers_generate_messages aruco_markers_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_cpp _aruco_markers_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_cpp _aruco_markers_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(aruco_markers_gencpp)
add_dependencies(aruco_markers_gencpp aruco_markers_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS aruco_markers_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/aruco_markers
)
_generate_msg_eus(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg"
  "${MSG_I_FLAGS}"
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/aruco_markers
)

### Generating Services

### Generating Module File
_generate_module_eus(aruco_markers
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/aruco_markers
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(aruco_markers_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(aruco_markers_generate_messages aruco_markers_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_eus _aruco_markers_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_eus _aruco_markers_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(aruco_markers_geneus)
add_dependencies(aruco_markers_geneus aruco_markers_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS aruco_markers_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/aruco_markers
)
_generate_msg_lisp(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg"
  "${MSG_I_FLAGS}"
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/aruco_markers
)

### Generating Services

### Generating Module File
_generate_module_lisp(aruco_markers
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/aruco_markers
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(aruco_markers_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(aruco_markers_generate_messages aruco_markers_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_lisp _aruco_markers_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_lisp _aruco_markers_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(aruco_markers_genlisp)
add_dependencies(aruco_markers_genlisp aruco_markers_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS aruco_markers_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/aruco_markers
)
_generate_msg_nodejs(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg"
  "${MSG_I_FLAGS}"
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/aruco_markers
)

### Generating Services

### Generating Module File
_generate_module_nodejs(aruco_markers
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/aruco_markers
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(aruco_markers_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(aruco_markers_generate_messages aruco_markers_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_nodejs _aruco_markers_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_nodejs _aruco_markers_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(aruco_markers_gennodejs)
add_dependencies(aruco_markers_gennodejs aruco_markers_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS aruco_markers_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/aruco_markers
)
_generate_msg_py(aruco_markers
  "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg"
  "${MSG_I_FLAGS}"
  "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/aruco_markers
)

### Generating Services

### Generating Module File
_generate_module_py(aruco_markers
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/aruco_markers
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(aruco_markers_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(aruco_markers_generate_messages aruco_markers_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/Marker.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_py _aruco_markers_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/air/swarm_ws/src/aruco_markers/msg/MarkerArray.msg" NAME_WE)
add_dependencies(aruco_markers_generate_messages_py _aruco_markers_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(aruco_markers_genpy)
add_dependencies(aruco_markers_genpy aruco_markers_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS aruco_markers_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/aruco_markers)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/aruco_markers
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(aruco_markers_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(aruco_markers_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/aruco_markers)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/aruco_markers
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(aruco_markers_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(aruco_markers_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/aruco_markers)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/aruco_markers
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(aruco_markers_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(aruco_markers_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/aruco_markers)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/aruco_markers
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(aruco_markers_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(aruco_markers_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/aruco_markers)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/aruco_markers\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/aruco_markers
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(aruco_markers_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(aruco_markers_generate_messages_py sensor_msgs_generate_messages_py)
endif()
