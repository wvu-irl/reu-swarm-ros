#include "ros/ros.h"
#include "wvu_swarm_std_msgs/robot_command_array.h"
#include "alice_swarm/Hub.h"
#include "alice_swarm/Robot.h"
#include "alice_swarm/aliceStructs.h"
#include <sstream>
#include <map>
void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
}
void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
}
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
}

int main(int argc, char **argv)
{

	std::map<int, Robot> aliceMap;

	ros::init(argc, argv, "AliceBrain");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("target", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("virtual_obstacles", 1000, obsCallback);
	ros::Rate loopRate(50);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::robot_command_array
			> ("final_execute", 1000);
	Hub aliceBrain(0); // Creates a hub for the conversion of absolute to relative info

	while (ros::ok())
	{

		wvu_swarm_std_msgs::vicon_points temp_target=
				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_points
						> ("virtual_targets"));
		wvu_swarm_std_msgs::vicon_bot_array temp_bot_array =
				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_bot_array
						> ("vicon_array"));
		wvu_swarm_std_msgs::vicon_points temp_obs_array =
				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_points
						> ("virtual_obstacles"));
		wvu_swarm_std_msgs::flows temp_flow_array =
						*(ros::topic::waitForMessage < wvu_swarm_std_msgs::flows
								> ("virtual_flows"));

		aliceBrain.update(temp_bot_array, temp_target, temp_obs_array,temp_flow_array); //puts in absolute data from subscribers
		for (int i = 0; i < temp_bot_array.poseVect.size(); i++)
		{
			aliceMap[i].receiveMsg(aliceBrain.getAliceMail(i)); //gives each robot the relative data it needs
		}
		std::vector<AliceStructs::ideal> all_ideals;
		for (std::map<int, Robot>::iterator it = aliceMap.begin();
				it != aliceMap.end(); ++it) //eventually run this part asynchronously
		{

			all_ideals.push_back(it->second.generateIdeal());
		}
		wvu_swarm_std_msgs::robot_command_array execute;
		for (std::map<int, Robot>::iterator it = aliceMap.begin();
				it != aliceMap.end(); ++it) //eventually run this part asynchronously
		{
			wvu_swarm_std_msgs::robot_command temp;
			AliceStructs::vel tempVel = it->second.generateComp(all_ideals);
			temp.rid = it->second.name;

			if (tempVel.mag>1) temp.r=1;
			else temp.r = tempVel.mag;
			temp.theta = 180/M_PI*fmod(2*M_PI+tempVel.dir,2*M_PI);

			execute.commands.push_back(temp);
		}

		pub.publish(execute);
		std::cout << "execute published" << std::endl;
		ros::spinOnce();
		loopRate.sleep();
	}
	return 0;
}
