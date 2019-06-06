#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>

#include <pthread.h>
#include <stdlib.h>
#include <vector>
#include <string>

// using a thread safe vector to collect data
static bool g_writing = false;
std::vector<wvu_swarm_std_msgs::robot_command> g_commands;

// struct for passing arguments to the threads
struct arg_struct
{
	ros::NodeHandle node;
	const int id;
	ros::MultiThreadedSpinner spinner;
};

// callback for subscription
void compressionCallback(wvu_swarm_std_msgs::robot_command msg)
{
	// spinning lock
	ros::Rate wait_rate(100);
	wait: while (g_writing)
	{
		wait_rate.sleep();
	}

	// writing
	if (!g_writing)
	{
		g_writing = true;
		g_commands.push_back(msg); // appending to the vector
	}
	else
	{
		goto wait;
	}
	g_writing = false;
}

// thread that use being used to collect robot commands
void *listeningThread(void *arg0)
{
	// thread start
	struct arg_struct *args = (struct arg_struct *) arg0; // unpacking parameters
	int id = args->id;
	ros::NodeHandle n = args->node;

	std::string topic; // constructing topic name
	char temp_topic_string[16] =
	{ '\0' };
	sprintf(temp_topic_string, "execute_%02d", id);
	topic = std::string(temp_topic_string);
	ROS_INFO("Subscribing to : %s", topic.c_str()); // checking
	ros::Subscriber exe = n.subscribe(topic, 1000, compressionCallback); // subscribing to topic
	ROS_INFO("Subscribed to  : %s", topic.c_str()); // checking

	ros::Rate t_rate(100);
	while (ros::ok())
	{
		args->spinner.spin();
		t_rate.sleep();
	}

}

// main
int main(int argc, char **argv)
{
	// initializing funnel node
	ros::init(argc, argv, "funnel");
	ros::NodeHandle n;

	// creating topic that will have the arrays published to it
	ros::Publisher fin_exe = n.advertise < wvu_swarm_std_msgs::robot_command_array
			> ("final_execute", 1000);

	ros::MultiThreadedSpinner spinner(50); // Makes a spinner that can handle each thread

	ros::Rate sub_rate(100);
	ros::Rate run_rate(100);

	// creating all the threads that will subscribe to the execution nodes
	for (size_t i = 0; i < 50; i++)
	{
		// creating a struct
		struct arg_struct args =
		{ n, (int) i, spinner };
		// creating a thread to subscribe to the robot topics
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_t tid;
		pthread_create(&tid, &attr, listeningThread, &args);

		sub_rate.sleep();
	}

	// collecting commands and packaging them for the
	while (ros::ok())
	{
		if (g_commands.size() > 10) // collecting data when there are at least 10 stored commands
		{
			wvu_swarm_std_msgs::robot_command_array cmds; // creating message to TCP server
			for (size_t i = 0; i < g_commands.size(); i++) // copying data
			{
				cmds.commands.push_back(g_commands[i]);
			}
			g_commands.clear();
			fin_exe.publish(cmds); // sending to server
		}

		run_rate.sleep();
	}

	return 0;
}
