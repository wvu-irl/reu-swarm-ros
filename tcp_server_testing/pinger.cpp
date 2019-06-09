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

#define PORT 4321
#define IP "10.0.1.37"

#define LATENCY_ONLY 1

using namespace std::chrono;

bool die = false;
auto start_time = high_resolution_clock::now();
double max_late = 0, prev = 0;

struct command
{
	char str[32];
};

void sigCallback(int sig)
{
	puts("\033[32mKilling process\033[0m");
	die = true;
}

int main()
{
	puts("\033[37;42mStarting\033[0m");

	signal(SIGINT, sigCallback);

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

	int bytes = write(socket_descriptor, "register YY", 11);
	if (bytes < 0)
	{
		puts("Failed registration");
		return 1;
	}
	puts("\033[1;34mRegistration successful\033[0m");

	while (true)
	{
		int message_size = 0;
		char str[32];

		usleep(1000);
		write(socket_descriptor, "ping", 4);
						start_time = high_resolution_clock::now();

		//reading the message
		while ((message_size = read(socket_descriptor, str, sizeof(str))) > 0)
		{
			if (strstr(str, "pong") == str)
			{
				double late = duration_cast<microseconds>(
						high_resolution_clock::now() - start_time).count() / 2000.0;
				bool new_max = false;
				if (max_late < late)
				{
					max_late = late;
					new_max = true;
				}
				printf("Latency: %s%4.3lf\033[0m\tmax:%s%4.3lf ms\033[0m\n", prev >= late ? "\033[32m" : "\033[31m" ,late, new_max ? "\033[30;41m" : "\033[0m", max_late);
				prev = late;
				usleep(500000);
				write(socket_descriptor, "ping", 4);
				start_time = high_resolution_clock::now();
				
			}
#if !LATENCY_ONLY
			else
				printf("CLIENT GOT: %s\n", str);
#endif
			// Display if there was an error
			if (message_size == -1)
			{
				puts("Error receiving message.");
				return 1;
			}

			if (die)
			{
				puts("Sending exit command");
				write(socket_descriptor, "exit", 4);
				exit(0);
			}
		}
		if (die)
		{
			puts("Sending exit command");
			write(socket_descriptor, "exit", 4);
			exit(0);
		}
		write(socket_descriptor, "ping", 4);
		start_time = high_resolution_clock::now();
		usleep(500000);
	}
	puts("\033[37;44mEnding\033[0m");
	return 0;
}
