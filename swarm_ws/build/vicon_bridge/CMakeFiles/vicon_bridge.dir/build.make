# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/air/swarm_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/air/swarm_ws/build

# Include any dependencies generated for this target.
include vicon_bridge/CMakeFiles/vicon_bridge.dir/depend.make

# Include the progress variables for this target.
include vicon_bridge/CMakeFiles/vicon_bridge.dir/progress.make

# Include the compile flags for this target's objects.
include vicon_bridge/CMakeFiles/vicon_bridge.dir/flags.make

vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o: vicon_bridge/CMakeFiles/vicon_bridge.dir/flags.make
vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o: /home/air/swarm_ws/src/vicon_bridge/src/vicon_bridge.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/air/swarm_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o"
	cd /home/air/swarm_ws/build/vicon_bridge && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o -c /home/air/swarm_ws/src/vicon_bridge/src/vicon_bridge.cpp

vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.i"
	cd /home/air/swarm_ws/build/vicon_bridge && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/air/swarm_ws/src/vicon_bridge/src/vicon_bridge.cpp > CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.i

vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.s"
	cd /home/air/swarm_ws/build/vicon_bridge && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/air/swarm_ws/src/vicon_bridge/src/vicon_bridge.cpp -o CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.s

vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.requires:

.PHONY : vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.requires

vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.provides: vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.requires
	$(MAKE) -f vicon_bridge/CMakeFiles/vicon_bridge.dir/build.make vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.provides.build
.PHONY : vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.provides

vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.provides.build: vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o


# Object files for target vicon_bridge
vicon_bridge_OBJECTS = \
"CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o"

# External object files for target vicon_bridge
vicon_bridge_EXTERNAL_OBJECTS =

/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: vicon_bridge/CMakeFiles/vicon_bridge.dir/build.make
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /home/air/swarm_ws/devel/lib/libmsvc_bridge.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /home/air/swarm_ws/src/vicon_bridge/vicon_sdk/Vicon_SDK_1.3_Linux/64-bit/libViconDataStreamSDK_CPP.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libdynamic_reconfigure_config_init_mutex.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libtf.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libtf2_ros.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libactionlib.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libmessage_filters.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libtf2.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libroscpp.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/librosconsole.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libxmlrpcpp.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libroscpp_serialization.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/librostime.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /opt/ros/kinetic/lib/libcpp_common.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge: vicon_bridge/CMakeFiles/vicon_bridge.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/air/swarm_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge"
	cd /home/air/swarm_ws/build/vicon_bridge && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vicon_bridge.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vicon_bridge/CMakeFiles/vicon_bridge.dir/build: /home/air/swarm_ws/devel/lib/vicon_bridge/vicon_bridge

.PHONY : vicon_bridge/CMakeFiles/vicon_bridge.dir/build

vicon_bridge/CMakeFiles/vicon_bridge.dir/requires: vicon_bridge/CMakeFiles/vicon_bridge.dir/src/vicon_bridge.cpp.o.requires

.PHONY : vicon_bridge/CMakeFiles/vicon_bridge.dir/requires

vicon_bridge/CMakeFiles/vicon_bridge.dir/clean:
	cd /home/air/swarm_ws/build/vicon_bridge && $(CMAKE_COMMAND) -P CMakeFiles/vicon_bridge.dir/cmake_clean.cmake
.PHONY : vicon_bridge/CMakeFiles/vicon_bridge.dir/clean

vicon_bridge/CMakeFiles/vicon_bridge.dir/depend:
	cd /home/air/swarm_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/air/swarm_ws/src /home/air/swarm_ws/src/vicon_bridge /home/air/swarm_ws/build /home/air/swarm_ws/build/vicon_bridge /home/air/swarm_ws/build/vicon_bridge/CMakeFiles/vicon_bridge.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vicon_bridge/CMakeFiles/vicon_bridge.dir/depend

