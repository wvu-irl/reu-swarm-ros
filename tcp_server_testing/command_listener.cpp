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

/*
 * this program registers with the server as a monitor
 * so, this means that all the commands that are being sent
 * are also sent to the monitor.
 *
 * This is a debugging tool for the TCP server
 */

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>
#include <errno.h>

#include <iostream>

#define PORT 4321
#define IP "192.168.10.187"

struct command
{
	char str[32];
};

int main()
{
	int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_descriptor == -1)
	{
		printf("Error getting socket: %d\n", errno);
		exit(1);
	}

	// Instantiate struct to store socket settings.
	struct sockaddr_in socket_address;
	memset(&socket_address, 0, sizeof(socket_address));
	socket_address.sin_family = AF_INET;
	socket_address.sin_port = htons(PORT);
	inet_aton(IP, &(socket_address.sin_addr));

	// Connect to the socket
	if (connect(socket_descriptor, (struct sockaddr*) &socket_address,
			sizeof(socket_address)) == -1)
	{
		printf("Error connecting to socket: %d\n", errno);
		exit(1);
	}

	struct command reg;
	for (size_t i = 0; i < sizeof(reg.str) / sizeof(char); i++)
	{

	}
	strcpy(reg.str, "register MN");
	int bytes = write(socket_descriptor, (&reg), sizeof(reg));
	if (bytes < 0)
	{
		puts("Failed registration");
		return 1;
	}

	while (true)
	{
		std::cout << "Top of loop" << std::endl;
		int message_size = 0;
		char str[32];

		int count = 0;

		//reading the message
		while ((message_size = read(socket_descriptor, str, sizeof(str))) > 0)
		{
			std::cout << "Message Recieved" << std::endl;
			printf("CLIENT GOT: %s\n", str);
			//sleep(1);
		}

		// Display if there was an error
		if (message_size == -1)
		{
			puts("Error receiving message.");
			return 1;
		}
//		sleep(1);
	}
	return 0;
}
