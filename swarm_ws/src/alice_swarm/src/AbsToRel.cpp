#include <ros/ros.h>
#include "alice_swarm/Hub.h"
#include <sstream>
#include <map>
#include <chrono>

//made global because i need them for the call back
//bool is_updated(false);
wvu_swarm_std_msgs::vicon_bot_array temp_bot_array;
wvu_swarm_std_msgs::map_levels temp_map;
wvu_swarm_std_msgs::vicon_points temp_target;
wvu_swarm_std_msgs::flows temp_flow_array;
wvu_swarm_std_msgs::chargers temp_charger_array;
wvu_swarm_std_msgs::energy temp_energy;
bool first = true;

void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
		temp_bot_array = msg;
}
void mapCallback(const wvu_swarm_std_msgs::map_levels &msg)
{
		temp_map= msg;
}
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
		temp_target = msg;
}
void flowCallback(const wvu_swarm_std_msgs::flows &msg)
{
		temp_flow_array = msg;
}
void chargerCallback(const wvu_swarm_std_msgs::chargers &msg)
{
	temp_charger_array = msg;
}
void energyCallback(const wvu_swarm_std_msgs::energy &msg)
{
	temp_energy= msg;
}

int main(int argc, char **argv)
{
//	if(first)
//	{
//		sleep(3);
//		first = false;
//	}
	//Creates an alice_hub node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AbsToRel");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("virtual_targets", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("map_data",1000,mapCallback);
	ros::Subscriber sub4 = n.subscribe("virtual_flows", 1000, flowCallback);
	ros::Subscriber sub5 = n.subscribe("chargers", 1000, chargerCallback);
	ros::Subscriber sub6 = n.subscribe("energy", 1000, energyCallback);
	ros::Rate loopRate(200);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::alice_mail_array> ("alice_mail_array", 1000);
	Hub alice_hub(0); // Creates a hub for the conversion of absolute to relative info
	while (ros::ok())
	{
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of hub

		alice_hub.update(temp_bot_array, temp_target, temp_map, temp_flow_array, temp_charger_array,
				temp_energy); //puts in absolute data from subscribers

		wvu_swarm_std_msgs::alice_mail_array mail = alice_hub.getAliceMail();

		mail.abs_chargers = temp_charger_array.charger;

		pub.publish(mail);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);

		//std::cout << "Time taken by Alice: " << duration.count() << " microseconds" << std::endl;
		//is_updated = true;
		ros::spinOnce();
		loopRate.sleep();

	}
	return 0;
}
