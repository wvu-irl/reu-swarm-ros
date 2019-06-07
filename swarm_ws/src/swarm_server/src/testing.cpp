#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>

#include <stdlib.h>

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

	ros::Publisher exe = n.advertise< wvu_swarm_std_msgs::robot_command_array >("final_execute", 1000);

	// creating recurrent message
	wvu_swarm_std_msgs::robot_command_array ary;
	ary.commands.push_back(genCmd("WV", 0.3f, 123.10f));
	ary.commands.push_back(genCmd("DE", 0.2f, 456.11f));
	ary.commands.push_back(genCmd("NH", 0.1f, 789.12f));

	ros::Rate rate(3);
	while (ros::ok())
	{
		exe.publish(ary);
		rate.sleep();
	}
}
