#include <ros/ros.h>
#include "alice_swarm/Hub.h"
#include <sstream>
#include <map>
#include <chrono>

//the various messages that get the hub must read and convert
wvu_swarm_std_msgs::vicon_bot_array temp_bot_array;
wvu_swarm_std_msgs::map_levels temp_map;
wvu_swarm_std_msgs::vicon_points temp_target;
wvu_swarm_std_msgs::flows temp_flow_array;
wvu_swarm_std_msgs::chargers temp_charger_array;
wvu_swarm_std_msgs::energy temp_energy;
wvu_swarm_std_msgs::sensor_data temp_sd;

//takes the positions of all of the robots
void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
	temp_bot_array = msg;
}

//takes the contour map
void mapCallback(const wvu_swarm_std_msgs::map_levels &msg)
{
		temp_map= msg;
}

//takes the positions of all the targets
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
		temp_target = msg;
}

//takes all of the flows
void flowCallback(const wvu_swarm_std_msgs::flows &msg)
{
		temp_flow_array = msg;
}

//takes the positions of all of the charging locations
void chargerCallback(const wvu_swarm_std_msgs::chargers &msg)
{
	temp_charger_array = msg;
}

//takes the energy of all of the robots
void energyCallback(const wvu_swarm_std_msgs::energy &msg)
{
	temp_energy = msg;
}

//takes the sensor data collected by the robots
void sensorCallback(const wvu_swarm_std_msgs::sensor_data &msg)
{
	temp_sd = msg;
}

int main(int argc, char **argv)
{
	//Creates an ABSolute To RELative conversion node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AbsToRel");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("virtual_targets", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("map_data",1000,mapCallback);
	ros::Subscriber sub4 = n.subscribe("virtual_flows", 1000, flowCallback);
	ros::Subscriber sub5 = n.subscribe("chargers", 1000, chargerCallback);
//	ros::Subscriber sub6 = n.subscribe("energy", 1000, energyCallback);
	ros::Subscriber sub7 = n.subscribe("sensor_data", 1000, sensorCallback);

	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::alice_mail_array> ("alice_mail_array", 1000);

	ros::Rate loopRate(10); //sets the publishing rate

	Hub alice_hub(0); // Creates a hub object for the conversion

	while (ros::ok()) //while ros is running
	{
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of hub

		alice_hub.update(temp_bot_array, temp_target, temp_map, temp_flow_array, temp_charger_array,
				temp_energy,temp_sd); //puts absolute data in hub from subscribers

		wvu_swarm_std_msgs::alice_mail_array mail = alice_hub.getAliceMail(); //calls Hub to convert all the information to a vector of

		mail.abs_chargers = temp_charger_array.charger; //the absolute positions are passed in directly.

		pub.publish(mail);

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);
#if DEBUG_HUB
		std::cout << "Time taken by AbsToRel " << duration.count() << " microseconds" << std::endl;
#endif
		ros::spinOnce();
		loopRate.sleep(); //sleep until rate

	}
	return 0;
}
