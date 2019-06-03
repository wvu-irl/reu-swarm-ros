#include <ros/ros.h>
#include <server_setup/robotcommand.h>
#include <server_setup/sensor_data.h>

#include "../arduino_server.h"
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

void *clientThread(void *arg)
{
	ROS_INFO("Thread start");
	int id = *((int *) arg);
	ROS_INFO("Client starting");
	int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_descriptor == -1) {
		puts("Error getting socket.");
		exit(1);
	}

	// Instantiate struct to store socket settings.
	struct sockaddr_in socket_address;
	memset(&socket_address, 0, sizeof(socket_address));
	socket_address.sin_family = AF_INET;
	socket_address.sin_port = htons(4321);
	inet_aton("127.0.0.1", &(socket_address.sin_addr));

	// Connect to the socket
	if (connect(socket_descriptor, (struct sockaddr *) &socket_address,
			sizeof(socket_address)) == -1) {
		puts("Error connecting to socket.");
		exit(1);
	}

	ROS_INFO("Connected to socket");

	command reg = { { '\0' } };
	sprintf(reg.str, "register %d", id);
	sleep(1);

	int bytes = write(socket_descriptor, (command *) (&reg), sizeof(command));
	if (bytes < 0) {
		puts("CLIENT ERROR: Failed registration");
		pthread_exit(NULL);
	}

	ROS_INFO("Client registered as %d", id);

	while (true) {
		// TODO Recieve
		int message_size = 0;
		command buffer = { { '\0' } };

		int count = 0;

		//reading the message
		while ((message_size = read(socket_descriptor, &buffer, sizeof(buffer)))
				> 0) {
			ROS_INFO("CLIENT %d GOT: %s", id, buffer.str);
		}

		// Display if there was an error
		if (message_size == -1) {
			ROS_INFO("Error receiving message.");
			exit(1);
		}
	}

	pthread_exit(0);
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "ros_server_tester");
	ros::NodeHandle n;

	ROS_INFO("Started");

	ros::Publisher send = n.advertise < server_setup::robotcommand
			> ("execute", 1000);

  ROS_INFO("Advertised publisher \"execute\"");

	std::vector < pthread_t > tids;

	ROS_INFO("Starting clients");
	int registers[] = {0, 1, 2, 3, 4, 5};
	for (size_t i = 0; i < sizeof(registers) / sizeof(int); i++) {
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_t tid;
		tids.push_back(tid);
		pthread_create(&tid, &attr, clientThread, (void *)(registers + i));
	}

	sleep(4);

	ROS_INFO("Creating command");
	server_setup::robotcommand test_send;
	test_send.rid = "DE";
	test_send.r = 4.0f;
	test_send.theta = 234.0f;

	ROS_INFO("Publishing command");
	send.publish(test_send);
	ROS_INFO("Published command");
	ros::spinOnce();

	ROS_INFO("Server waiting for clients");
	//waiting for all clients to die
	sleep(10);

	return 0;
}
