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


void *clientThread(void *arg)
{


	pthread_exit(0);
}


int main(int argc, char **argv) {
	ros::init(argc, argv, "ros_server_tester");
	ros::NodeHandle n;

	ros::Publisher send = n.advertise < server_setup::robotcommand
			> ("execute", 1000);

	std::vector<pthread_t> tids;

	for (size_t i = 0; i < 3; i++) {
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_t tid;
		tids.push_back(tid);
		pthread_create(&tid, &attr, clientThread, NULL);
	}

	// TODO do ROS stuff

	for (pthread_t tid : tids)
	{
		pthread_join(tid);
	}

	return 0;
}
