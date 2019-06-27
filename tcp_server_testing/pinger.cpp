/**
 *
 * Pinger is a program that can measure the latency between server and client
 *
 * The time is measured in ms and measures the round trip of the packet.
 * Program pings server at 2Hz roughly.
 *
 *
 * author: Henry Vos
 * email: hlv0002@mix.wvu.edu
 *
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
#include <signal.h>

#include <chrono>
#include <fstream>

#define PORT 4321

// ip that the program is connecting to
#define IP "192.168.10.187"

// using this namespace because there are many time measurments
using namespace std::chrono;

// flag for SIGINT
bool die = false;

// time measurement
auto start_time = high_resolution_clock::now();

// variables for doing some basic visual heuristics
double max_late = 0, prev = 0;

// callback for SIGINT
void sigCallback(int sig)
{
	puts("\033[32mKilling process\033[0m");
	die = true;
}

// main
int main()
{
	puts("\033[37;42mStarting\033[0m");

	// overriding interrupt signal
	signal(SIGINT, sigCallback);

	// setting up socket from the client side
	int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_descriptor == -1)
	{
		printf("\033[30;41mError getting socket: %d\033[0m\n", errno);
		exit(1);
	}
	puts("Got socket");

	puts("Started thread");

	// Instantiate struct to store socket settings.
	struct sockaddr_in socket_address;
	memset(&socket_address, 0, sizeof(socket_address));
	socket_address.sin_family = AF_INET;
	socket_address.sin_port = htons(PORT);
	inet_aton(IP, &(socket_address.sin_addr));

	// Connect to the socket
	if (connect(socket_descriptor, (struct sockaddr *) &socket_address,
			sizeof(socket_address)) == -1)
	{
		printf("Error connecting to socket: %d\n", errno);
		exit(1);
	}
	puts("\033[37;42mConnected to socket\033[0m");

	int count = 0;

	// main loop
	while (count < 200000)
	{
		int message_size = 0;
		char str[32];

		write(socket_descriptor, "ping", 4);
		start_time = high_resolution_clock::now();

		//reading the message
		while ((message_size = read(socket_descriptor, str, sizeof(str))) > 0 && count < 200000)
		{
			// looking for pong return
			if (strstr(str, "pong") == str)
			{
				// calculating latency
				double late = duration_cast < microseconds
						> (high_resolution_clock::now() - start_time).count() / 1000.0;
				bool new_max = false; // determining if the max recorded latency has increased
				if (max_late < late)
				{
					max_late = late;
					new_max = true;
				}
				// writing to file
				std::ofstream file;
				file.open("latency_log.csv", std::ios::out | std::ios::app);
				file << late << std::endl;
				file.close();
				count++;
				// displaying latency results
				printf("[%07d]Latency: %s%4.3lf\033[0m\tmax:%s%4.3lf ms\033[0m\n", count,
						prev >= late ? "\033[32m" : "\033[31m", late,
						new_max ? "\033[30;41m" : "\033[0m", max_late);
				prev = late;
				// sleeping
				//usleep(500000);
				write(socket_descriptor, "ping", 4); // sending a ping
				start_time = high_resolution_clock::now(); // resetting start time

			}
			// Display if there was an error
			if (message_size == -1)
			{
				puts("Error receiving message.");
				return 1;
			}

			// runs if there was an interrupt
			if (die)
			{
				puts("Sending exit command");
				write(socket_descriptor, "exit", 4);
				exit(0);
			}
		}

		// this runs when there are no messages coming through
		// runs on interrupt
		if (die)
		{
			puts("Sending exit command");
			write(socket_descriptor, "exit", 4);
			exit(0);
		}

		// writes another ping if missed
		write(socket_descriptor, "ping", 4);
		start_time = high_resolution_clock::now();
		//usleep(500000); // sleeping 0.5s
	}
	puts("\033[37;44mEnding\033[0m");
	return 0;
}
