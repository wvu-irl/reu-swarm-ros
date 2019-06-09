#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>

#include <string.h>

#include <stdlib.h>
#include <sys/types.h>
#include <pthread.h>

#define SERVER_TEST 1
#define FUNNEL_TEST 0

#if FUNNEL_TEST && !SERVER_TEST

struct arg_pass
{
	ros::NodeHandle *nh;
	int id;
};

void *sendingThread(void *arg0)
{
	struct arg_pass *args = (struct arg_pass *) arg0;
	char name[16] = { '\0' };
	sprintf(name, "execute_%02d", args->id);
	ros::Publisher pub = args->nh->advertise < wvu_swarm_std_msgs::robot_command
			> (name, 1000);

	ROS_INFO("\033[1;34mPublishing to: %s", name);

	wvu_swarm_std_msgs::robot_command cmd;
	cmd.rid[0] = '0' + (args->id / 10);
	cmd.rid[1] = '0' + (args->id % 10);
	cmd.r = (float) args->id;
	cmd.theta = 10.0f * (float) args->id;

	while (ros::ok())
	{
		//ROS_INFO("\033[34m[%d]: Publishing", args->id);
		pub.publish(cmd);
		usleep(10000);
	}
}

void subCallback(wvu_swarm_std_msgs::robot_command_array opt)
{
	ROS_INFO("\033[30;42mGot message\033[0m");
	for (int i = 0; i < opt.commands.size(); i++)
	{
		ROS_INFO("\033[32m[%02d]: RID:%c%c R:%0.3f TH:%0.3f\033[0m", i,
				opt.commands[i].rid[0], opt.commands[i].rid[1], opt.commands[i].r,
				opt.commands[i].theta);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "testing");
	ros::NodeHandle n;

	struct arg_pass args[20];

	for (int i = 0; i < sizeof(args) / sizeof(struct arg_pass); i++)
	{
		args[i] = (struct arg_pass ) { &n, (int) i };
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_t tid;
		pthread_create(&tid, &attr, sendingThread, args + i);
	}

	sleep(3);

	n.subscribe("final_execute", 1000, subCallback);
	ROS_INFO("\033[1;34mSubscribed to /final_execute");
	ros::spin();
	return 0;
}

#endif

#if SERVER_TEST && !FUNNEL_TEST

wvu_swarm_std_msgs::robot_command genCmd(const char id[3], float r, float theta)
{
	wvu_swarm_std_msgs::robot_command cmd;
	boost::array<unsigned char, 2> aid;
	aid[0] = id[0];
	aid[1] = id[1];
	cmd.rid = aid;
	cmd.r = r;
	cmd.theta = theta;
	return cmd;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "testing");
	ros::NodeHandle n;

	srand(20473598);

	ros::Publisher exe = n.advertise < wvu_swarm_std_msgs::robot_command_array
			> ("final_execute", 1000);

	// creating recurrent message
	sleep(6);
	ROS_WARN("Sending messages");

	ros::Rate rate(3);
	while (ros::ok())
	{
		wvu_swarm_std_msgs::robot_command_array ary;
		for (int i = 0; i < 10; i++)
		{
			char nm[3] = { '\0' };
			sprintf(nm, "%02d", rand() % 50);
			ary.commands.push_back(
					genCmd(nm, (float) (rand() % 100) / 100.0f,
							(float) (rand() % 3600) / 10.0f));
		}

		exe.publish(ary);
		rate.sleep();
	}
}
#endif
