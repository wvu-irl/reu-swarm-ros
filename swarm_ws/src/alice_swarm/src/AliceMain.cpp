#include "ros/ros.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/robot_command_array.h"
#include "alice_swarm/Hub.h"
#include "alice_swarm/Alice.h"
#include "alice_swarm/aliceStructs.h"
#include <sstream>
#include <map>
#include <chrono>

wvu_swarm_std_msgs::alice_mail_array temp_mail;

void aliceCallback(const wvu_swarm_std_msgs::alice_mail_array &msg)
{
	temp_mail=msg;
}

int main(int argc, char **argv)
{
	std::map<int, Alice> aliceMap; //Maps robot id's to robots so they can be accessed easily

	//Creates an AliceBrain node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AliceBrain");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("alice_mail_array", 1000, aliceCallback);
	ros::Rate loopRate(10);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::robot_command_array > ("final_execute", 1000);

	while (ros::ok())
	{
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of Alice
		for (int i = 0; i < temp_mail.mails.size(); i++)
		{
			aliceMap[temp_mail.mails.at(i).name].updateModel(temp_mail.mails.at(i)); //gives each robot the relative data it needs
		}
		wvu_swarm_std_msgs::robot_command_array execute;
		for (std::map<int, Alice>::iterator it = aliceMap.begin(); it != aliceMap.end(); ++it) //eventually run this part asynchronously
		{
			wvu_swarm_std_msgs::robot_command temp;
			AliceStructs::vel tempVel = it->second.generateVel(); //Uses all of the ideals to generate compromises
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
