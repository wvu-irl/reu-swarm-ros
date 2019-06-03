#include <ros/ros.h>

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
#include <pthread>

void *clientThread(void *arg) {
	int id = *((int *) arg)
	sleep(2);

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

	command reg = { { '\0' } };
	sprintf(reg.str, "register %d", id);

	sleep(1);

	int bytes = write(socket_descriptor, (command *) (&reg), sizeof(command));
	if (bytes < 0) {
		puts("CLIENT ERROR: Failed registration");
		return;
	}

	while (true) {
		// TODO Recieve
		int message_size = 0;
		command buffer = { { '\0' } };

		int count = 0;

		//reading the message
		while ((message_size = read(socket_descriptor, &buffer, sizeof(buffer)))
				> 0) {
			printf("CLIENT %d GOT: %s\n", id, buffer.str);
		}

		// Display if there was an error
		if (message_size == -1) {
			puts("Error receiving message.");
			exit(1);
		}
	}

	pthread_exit(0);
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "ros_server_tester");
	ros::NodeHandle n;

	ros::Publisher send = n.advertise < server_setup::robotcommand
			> ("execute", 1000);

	std::vector < pthread_t > tids;

	for (size_t i = 0; i < 3; i++) {
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_t tid;
		tids.push_back(tid);
		pthread_create(&tid, &attr, clientThread, i);
	}

	sleep(1);

	server_setup::robotcommand test_send;
	test_send.rid = 0;
	test_send.r = 4.0f;
	test_send.theta = 234.0f;

	send.publish(test_send);

	// waiting for all clients to die
	for (pthread_t tid : tids) {
		pthread_join(tid);
	}

	return 0;
}
