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

	//Creates an AliceBrain node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AliceBrain");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("virtual_targets", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("virtual_obstacles", 1000, obsCallback);
	ros::Subscriber sub4 = n.subscribe("virtual_flows", 1000, flowCallback);
	ros::Rate loopRate(10);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::robot_command_array > ("final_execute", 1000);
	Hub aliceBrain(0); // Creates a hub for the conversion of absolute to relative info

	while (ros::ok())
	{
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of Alice

		aliceBrain.update(temp_bot_array, temp_target, temp_obs_array, temp_flow_array); //puts in absolute data from subscribers
		for (int i = 0; i < temp_bot_array.poseVect.size(); i++)
		{

			aliceMap[i].receiveMsg(aliceBrain.getAliceMail(i)); //gives each robot the relative data it needs
		}
		std::vector<AliceStructs::ideal> all_ideals; //Creates a vector that stores the robot's initial ideal vectors
		for (std::map<int, Robot>::iterator it = aliceMap.begin(); it != aliceMap.end(); ++it) //eventually run this part asynchronously
		{

			all_ideals.push_back(it->second.generateIdeal()); //Adds the ideals to the vector
		}
		wvu_swarm_std_msgs::robot_command_array execute;
		for (std::map<int, Robot>::iterator it = aliceMap.begin(); it != aliceMap.end(); ++it) //eventually run this part asynchronously
		{
			wvu_swarm_std_msgs::robot_command temp;
			AliceStructs::vel tempVel = it->second.generateComp(all_ideals); //Uses all of the ideals to generate compromises
			temp.rid = it->second.name;

			if (tempVel.mag > 1) //caps the speed
				temp.r = 1;
			else
				temp.r = tempVel.mag;
			temp.theta = 180 / M_PI * fmod(2 * M_PI + tempVel.dir, 2 * M_PI);

			execute.commands.push_back(temp); //adds vector to published vector
		}

		pub.publish(execute);
//		std::cout << "execute published" << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);

		//std::cout << "Time taken by Alice: " << duration.count() << " microseconds" << std::endl;
		//is_updated = true;
		ros::spinOnce();
		loopRate.sleep();

	}
	return 0;
}
