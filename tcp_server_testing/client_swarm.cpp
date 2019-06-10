/*
 * client_swarm.cpp
 *
 *  Created on: Jun 6, 2019
 *      Author: Henry Vos
 *
 *  This is meant to simulate connections to the server
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

#include "robot_id.h"

#define PORT 4321
#define IP "127.0.0.1"

struct command
{
	char str[32];
};

int main()
{
	size_t i;
	pid_t curr_pid = getpid();
	for (i = 0; i < 35 && curr_pid != 0; i++)
	{
		curr_pid = fork();
		usleep(100000);
	}
	if (curr_pid == 0)
		i++;

	int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_descriptor == -1)
	{
		printf("Error getting socket: %d\n", errno);
		if (curr_pid != 0)
			wait(NULL);
		exit(1);
	}

	// Instantiate struct to store socket settings.
	struct sockaddr_in socket_address;
	memset(&socket_address, 0, sizeof(socket_address));
	socket_address.sin_family = AF_INET;
	socket_address.sin_port = htons(PORT);
	inet_aton(IP, &(socket_address.sin_addr));

	// Connect to the socket
	if (connect(socket_descriptor, (struct sockaddr *)&socket_address,
				sizeof(socket_address)) == -1)
	{
		printf("\033[1;31mError connecting to socket: %d\033[0m\n", errno);
		if (curr_pid != 0)
			wait(NULL);
		exit(1);
	}

	struct command reg;
	for (size_t i = 0; i < sizeof(reg.str) / sizeof(char); i++)
	{
		reg.str[i] = '\0';
	}
	sprintf(reg.str, "register %s", rid_indexing[i].c_str());
	int bytes = write(socket_descriptor, (&reg), sizeof(reg));
	if (bytes < 0)
	{
		puts("Failed registration");
		if (curr_pid != 0)
			wait(NULL);
		return 1;
	}
	puts("");
	std::cout << "\033[1;32mRegistered as " << reg.str << "\033[0m"
			  << std::endl;

	while (true)
	{
		int message_size = 0;
		char str[32];

		int count = 0;

		//reading the message
		while ((message_size = read(socket_descriptor, str, sizeof(str))) > 0)
		{
			printf("CLIENT %s GOT: %s\n", rid_indexing[i].c_str(), str);
		}

		// Display if there was an error
		if (message_size == -1)
		{
			std::cout << "\033[1;33mError recieving message: " << errno << "\033[0m" << std::endl;
			if (curr_pid != 0)
				wait(NULL);
			return 1;
		}
	}
	if (curr_pid != 0)
		wait(NULL);
	return 0;
}