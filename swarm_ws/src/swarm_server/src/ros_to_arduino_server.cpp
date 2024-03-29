/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
#include <wvu_swarm_std_msgs/sensor_data.h>

#include "arduino_server.h"
#include <swarm_server/robot_id.h>
#include <swarm_server/battery_states.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <sstream>
#include <string>
#include <functional>
#include <signal.h>

#include <iostream>
#include <fstream>

#define LOG_TO_CSV 0

#define DEBUG 0
#define PRINT_HEADER "[\033[1;33mros_to_arduino_server\033[0m]"

ros::Publisher g_from_ard; // global publisher for data from the arduinos

/**
 * This is a function that flaggs when an interrupt occurs
 * This is used when "ctrl+C" is used to kill a program
 */
volatile sig_atomic_t g_flag = 0;
void flagger(int sig)
{
	g_flag = 1;
//        system("fuser 4321/tcp -k");
}

/**
 * Function for keeping all loops and threads alive
 * Based on ros ok and the interrupt flag
 */
bool keepAlive()
{
	return !g_flag && ros::ok() && !g_server_failure;
}

/**
 *  Callback for getting data from the arduino clients
 *
 *  All recieved data will get published as soon as it comes
 */
void commandCallback(command cmd, int rid)
{
	ROS_INFO("\033[34mCommand: \033[30;44m%s\033[0m", cmd.str); // displaying recieved data
	wvu_swarm_std_msgs::sensor_data inf; // conversion container
	inf.rid = rid;
	inf.battery_state = NONE;
	inf.battery_level = -1;
	if (strcmp(cmd.str, "charged") == 0)
	{
		inf.battery_state = CHARGED;
	}
	else if (strcmp(cmd.str, "going") == 0)
	{
		inf.battery_state = GOING;
	}
	else if (strcmp(cmd.str, "charging") == 0)
	{
		inf.battery_state = CHARGING;
	}
	else if (strcmp(cmd.str, "error") == 0)
	{
		inf.battery_state = ERROR;
	}
	else
	{
		inf.battery_level = (float) strtod(cmd.str, NULL);
	}
	if (&g_from_ard != NULL)
		g_from_ard.publish(inf); // publishing

#if LOG_TO_CSV
	std::ofstream file;
	file.open("/home/air/sensor_log.csv", std::ios::out | std::ios::app);
	file << rid_indexing[rid] << "," << cmd.str << std::endl;
	file.close();
#endif
}

/**
 *  Info callback
 *
 *  Displays server information
 */
void info(const char *patt, void *dat)
{
	std::ostringstream os;
	os << "SERVER INFO: " << patt;
	std::string full_pattern = os.str();
	const char *ch_pathh = full_pattern.c_str();
	ROS_INFO(ch_pathh, dat);
}

/**
 *  callback for a subscription to send data to the swarm
 */
void sendToRobotCallback(wvu_swarm_std_msgs::robot_command_array msga)
{
	for (wvu_swarm_std_msgs::robot_command msg : msga.commands)
	{
		if (!keepAlive())
			break;
		command cmd = { { '\0' } }; // creating command
		sprintf(cmd.str, "%f,%f,%s", msg.r, msg.theta, msg.sys_comm.c_str());
		int id = msg.rid;
#if DEBUG
		ROS_INFO(PRINT_HEADER"Constructed command: %02d,\t%s", id, cmd.str);
#endif
		sendCommandToRobots(cmd, id); // sending to robots through TCP server
		if (!keepAlive())
			break;
	}
}

/**
 *  Callback to display server errors
 */
void errorCallBack(const char *msg)
{
	ROS_ERROR(msg, NULL);
}

/**
 *  Runs a loop separate from the server
 *
 *  This is nessessary because the server contains a closed loop
 */
void* controlThread(void *arg0)
{
	while (keepAlive())
	{
		ros::spinOnce();
#if DEBUG  
//               puts("Thread running");
#endif
	}
#if DEBUG  
	puts("\033[1;32mThread exiting\033[0m");
#endif
	pthread_exit(0); // exiting thread
									 // this is probably unreachable
}

void warnInfo(const char *str)
{
	ROS_WARN(str, NULL);
}

// main
int main(int argc, char **argv)
{
	ros::init(argc, argv, "arduino_server");
	ros::NodeHandle n;

	signal(SIGINT, flagger);

	ros::Subscriber to_ard = n.subscribe("final_execute", 1000,
			sendToRobotCallback); // subscribing to movment datastream

	g_from_ard = n.advertise < wvu_swarm_std_msgs::sensor_data
			> ("/sensor_data", 1000);

	ROS_INFO("Starting");
	ROS_INFO("Sending %d bytes per message\n", (int) sizeof(command));

	// creating a separate thread for additional looped control
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_t tid;
#if DEBUG
	ROS_INFO("Starting main-thread");
#endif
	pthread_create(&tid, &attr, controlThread, &n);
#if DEBUG
	ROS_INFO("Starting server");
#endif
	// starting server
	beginServer(commandCallback, info, errorCallBack, keepAlive, warnInfo);
#if DEBUG
	ROS_INFO("Waiting for main-thread to die");
#endif

	// waiting for thread to die
	pthread_join(tid, NULL);
	ROS_WARN("Server is \033[1;31mDEAD\033[0m");
	return 0;
}
