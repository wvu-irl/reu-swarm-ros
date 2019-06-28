#include "ros/ros.h"
#include "wvu_swarm_std_msgs/robot_command_array.h"
#include "alice_swarm/Hub.h"
#include "alice_swarm/Robot.h"
#include "alice_swarm/aliceStructs.h"
#include <sstream>
#include <map>
#include <chrono>

//made global because i need them for the call back
//bool is_updated(false);
wvu_swarm_std_msgs::vicon_bot_array temp_bot_array;
wvu_swarm_std_msgs::vicon_points temp_obs_array;
wvu_swarm_std_msgs::vicon_points temp_target;
wvu_swarm_std_msgs::flows temp_flow_array;

void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
		temp_bot_array = msg;
}
void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
		temp_obs_array = msg;
}
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
		temp_target = msg;
}
void flowCallback(const wvu_swarm_std_msgs::flows &msg)
{
		temp_flow_array = msg;
}


int main(int argc, char **argv)
{

	std::map<int, Robot> aliceMap; //Maps robot id's to robots so they can be accessed easily

	//Creates an alice_hub node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AbsToRel");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("virtual_targets", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("virtual_obstacles", 1000, obsCallback);
	ros::Subscriber sub4 = n.subscribe("virtual_flows", 1000, flowCallback);
	ros::Rate loopRate(15);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::alice_mail_array> ("alice_mail_array", 1000);
	Hub alice_hub(0); // Creates a hub for the conversion of absolute to relative info

	while (ros::ok())
	{
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of hub

		alice_hub.update(temp_bot_array, temp_target, temp_obs_array, temp_flow_array); //puts in absolute data from subscribers

		wvu_swarm_std_msgs::alice_mail_array mail = alice_hub.getAliceMail();

		pub.publish(mail);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);

//		std::cout << "Time taken by Alice: " << duration.count() << " microseconds" << std::endl;
		//is_updated = true;
		ros::spinOnce();
		loopRate.sleep();

	}
	return 0;
}
