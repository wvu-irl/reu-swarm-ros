#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>

#include "arduino_server.h"
#include <swarm_server/robot_id.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <sstream>
#include <string>
#include <functional>
#include <signal.h>

#include <iostream>
#include <fstream>

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
}

/**
 *  Callback for getting data from the arduino clients
 *
 *  All recieved data will get published as soon as it comes
 */
void commandCallback(command cmd, int rid)
{
	ROS_INFO("\033[34mCommand: \033[30;44m%s\033[0m", cmd.str); // displaying recieved data
//	wvu_swarm_std_msgs::sensor_data inf; // conversion container
//	for (size_t i = 0; i < 32; i++)
//	{
//		inf.data[i] = cmd.str[i];
//	}
//	if (&g_from_ard != NULL)
//		g_from_ard.publish(inf); // publishing
//
//	std::ofstream file;
//	file.open("/home/air/sensor_log.csv", std::ios::out | std::ios::app);
//	file << rid_indexing[rid] << "," << cmd.str << std::endl;
//	file.close();
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
#if DEBUG
		ROS_INFO(PRINT_HEADER"Sending message: %c%c, %f, %f", msg.rid[0],
				msg.rid[1], msg.r, msg.theta);
#endif
		command cmd = { { '\0' } }; // creating command
		sprintf(cmd.str, "%f,%f", msg.r, msg.theta);
		int id = msg.rid;
#if DEBUG
		ROS_INFO(PRINT_HEADER"Constructed command: %02d,\t%s", id, cmd.str);
#endif
		sendCommandToRobots(cmd, id); // sending to robots through TCP server
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
void *controlThread(void *arg0)
{
	signal(SIGINT, flagger);
	ros::NodeHandle *n = (ros::NodeHandle *) arg0; // passed node handle
	ros::Subscriber to_ard = n->subscribe("final_execute", 1000,
			sendToRobotCallback); // subscribing to movment datastream

	while (true)
	{
		if (g_flag || !ros::ok())
		{
			exit(0); // exiting program when it is supposed to
		}

		ros::spinOnce();
	}

	pthread_exit(0); // exiting thread
									 // this is probably unreachable
}

/**
 * Function for keeping all loops and threads alive
 * Based on ros ok and the interrupt flag
 */
bool keepAlive()
{
	return !g_flag;
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

	ROS_INFO("Starting");
	ROS_INFO("Sending %d bytes per message\n", (int) sizeof(command));

	// creating a separate thread for additional looped control
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_t tid;
	pthread_create(&tid, &attr, controlThread, &n);

	// starting server
	beginServer(commandCallback, info, errorCallBack, keepAlive, warnInfo);

	// waiting for thread to die
	pthread_join(tid, NULL);
	return 0;
}
