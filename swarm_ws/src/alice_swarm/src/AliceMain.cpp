#include "ros/ros.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/robot_command_array.h"
#include "wvu_swarm_std_msgs/map.h"
#include "alice_swarm/Hub.h"
#include "alice_swarm/Alice.h"
#include "alice_swarm/aliceStructs.h"
#include <alice_swarm/get_maps.h>

#include <sstream>
#include <map>
#include <chrono>

wvu_swarm_std_msgs::alice_mail_array temp_mail;

void aliceCallback(const wvu_swarm_std_msgs::alice_mail_array &msg)
{
	temp_mail = msg;
}

int main(int argc, char **argv)
{
	std::map<int, Alice> alice_map; //Maps robot id's to robots so they can be accessed easily
	std::map<int, Alice>::iterator map_it = alice_map.begin();
	//Creates an AliceBrain node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AliceBrain");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("alice_mail_array", 1000, aliceCallback);

	ros::Rate loopRate(50);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::robot_command_array > ("final_execute", 1000);
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::map > ("info_map", 1000);
	ros::ServiceClient client = n.serviceClient < alice_swarm::get_maps > ("get_maps");

	while (ros::ok())
	{
		// it!=mymap.end(); ++it)
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of Alice
		alice_swarm::get_maps srv;
		client.call(srv);

		std::vector<wvu_swarm_std_msgs::map> maps;
	 for (int i=0; i<srv.response.maps.size(); i++) maps.push_back(srv.response.maps.at(i));

	  std::vector<int> ids;
	  for (int i=0; i<srv.response.ids.size();i++) ids.push_back(srv.response.ids.at(i));

		for (int i = 0; i < temp_mail.mails.size(); i++)
		{
			alice_map[temp_mail.mails.at(i).name].updateModel(temp_mail.mails.at(i),maps,ids); //gives each robot the relative data it needs, whilst also creating the alice's
		}

		//One alice is selected to send her map out to the rest of them; less robots means a particular robot's map is sent more often
		if (alice_map.size() != 0)
		{
			if (map_it == alice_map.end() )
			{
				map_it = alice_map.begin();
				map_it->second.model.pass(pub2);
				map_it++;
			} else
			{
				map_it->second.model.pass(pub2);
				map_it++;
			}
		}

		wvu_swarm_std_msgs::robot_command_array execute;
		for (std::map<int, Alice>::iterator it = alice_map.begin(); it != alice_map.end(); ++it) //eventually run this part asynchronously
		{
			wvu_swarm_std_msgs::robot_command temp;
			AliceStructs::vel tempVel = it->second.generateVel(); //Uses all of the ideals to generate compromises and commands.
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

		std::cout << "Time taken by Alice: " << duration.count() << " microseconds" << std::endl;
		//is_updated = true;
		ros::spinOnce();
		loopRate.sleep();

	}
	return 0;
}
