#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>

#include <pthread.h>
#include <stdlib.h>
#include <vector>
#include <string>

static boost::array<wvu_swarm_std_msgs::robot_command, 50> g_commands;

struct arg_struct
{
	ros::NodeHandle node;
	int id;
};

void compressionCallback(wvu_swarm_std_msgs::robot_command msg)
{
	g_commands.push_back(msg);
}

void *listeningThread(void *arg0)
{
	struct arg_strict *args = (struct arg_struct *)arg0;
	int id = args->id;
	ros::NodeHandle n = args->node;
	std::string topic = "execute_" + id;
	ros::Subscriber exe = n.subscribe(topic, 1000, compressionCallback);
	ROS_INFO("Subscribing to : %s", topic);
	ros::spin();
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "topic_compressor");

	ros::NodeHandle n;

	ros::Publisher fin_exe = n.advertise< wvu_swarm_std_msgs::robot_command_array >("final_execute", 1000);

	for (size_t i = 0; i < g_commands.size(); i++)
	{
		struct arg_struct args;
		args.node = n;
		args.id = i;

		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_t tid;
		pthread_create(&tid, &attr, listeningThread, &args);
	}

	while (ros::ok())
	{
		if (g_commands.size() > 0)
		{
			wvu_swarm_std_msgs::robot_command_array cmds;
			for (size_t i = 0;i < g_commands.size();i++)
			{
				cmds.commands.push_back(g_commands.at(i));
			}

			fin_exe.publish(cmds);
		}
	}

	return 0;
}
