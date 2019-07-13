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
