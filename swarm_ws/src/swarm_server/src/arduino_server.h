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

#ifndef SERVER_HH
#define SERVER_HH

/**
 *
 * Server
 *
 * Author: Henry Vos
 *
 * Purpose:
 *   This section is to accept connections from all the robots.
 *   The clients will each get their own process. (this may pose issues)
 *	 The server will also pipe messages to the clients
 */

// Includes
#include <swarm_server/robot_id.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/wait.h>

#include <functional>
#include <vector>
#include <map>
#include <string>

#define SERVER_PORT 4321 // port number
#define COMMAND_SIZE 64

bool g_server_failure = false;

// datastructure that is used to send/recieve commands
// this is using a struct as it is only a public data storing application
typedef struct
{
	char str[COMMAND_SIZE];
} command;

/**
 *  Class contains all the data necessesary to talk with one robot
 *  as well as act as a filter with the rid to reduce bandwidth usage
 *
 */
class ConnectionInfo
{
private:
	int connection_descriptor; // the connection connection_descriptor for the socket
	int rid; // the id of the robot (or other client) connected

public:
	ConnectionInfo(int connection_descriptor);

	int getRID(); // returns the RID of the client
	void setRID(int rid); // sets the RID of the client

	int getConnectionDescriptor(); // returns the connection descriptor of the client
};

/**
 * Send command to robots sends commands to robots
 *
 * cmd is the content of the comannd
 *
 * recip_rid is the recipiant's RID
 *  special cases:
 *  a connection with registered id of -2 will recieve all sent commands
 *  a recip_rid of -1 will send a command to all robots
 */
void sendCommandToRobots(command cmd, int recip_rid);

// function responsible for recieving information from a client
void runClient(std::function<void(command, int)> command_callback,
		std::function<void(const char *, void *)> info_callback,
		std::function<void(const char *)> error_callback,
		std::function<bool()> exit_condition_callback, int id);

/**
 * Begins accepting connections to the server and processes commands from them
 *
 * command_callback is a registered function for taking care of command contents
 * info_callback is the registered function for printing basic information to the screen
 * error_callback is the registered function for showing server errors
 * exit_condition_callback is the registered function that dictates a successful exit condition
 * 													exits when exit_condition_callback() == false
 */
int beginServer(std::function<void(command, int)> command_callback,
		std::function<void(const char *, void *)> info_callback,
		std::function<void(const char *)> error_callback,
		std::function<bool()> exit_condition_callback,
		std::function<void(const char *)> warn_callback);
#include "arduino_server_source.cpp"
#endif
