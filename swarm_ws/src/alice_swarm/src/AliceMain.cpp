#include "ros/ros.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/robot_command_array.h"
#include <wvu_swarm_std_msgs/priorities.h>
#include "wvu_swarm_std_msgs/map.h"
#include "alice_swarm/Hub.h"
#include "alice_swarm/Alice.h"
#include "alice_swarm/aliceStructs.h"
#include <alice_swarm/get_maps.h>

#include <sstream>
#include <map>
#include <chrono>

wvu_swarm_std_msgs::alice_mail_array temp_mail;
wvu_swarm_std_msgs::priorities temp_priorities;
wvu_swarm_std_msgs::chargers prev_chargers;
bool first = true;
bool same;

void aliceCallback(const wvu_swarm_std_msgs::alice_mail_array &msg)
{
	temp_mail = msg;
}

void prioritiesCallback(const wvu_swarm_std_msgs::priorities &msg)
{
	temp_priorities = msg;
}

void checkForUpdatedChargers()
{
	if(first)
	{
		prev_chargers.charger = temp_mail.abs_chargers;
		first = false;
	}
	else
	{
		int i = 0;
		same = true;
		while(i < prev_chargers.charger.size() && same)
		{
			if(prev_chargers.charger.at(i).occupied != temp_mail.abs_chargers.at(i).occupied)
			{
				same = false;
			}
			i++;
		}
		if(same)
		{
		}
		else
		{
			temp_mail.abs_chargers = prev_chargers.charger;
		}
	}
}

int main(int argc, char **argv)
{
	std::map<int, Alice> alice_map; //Maps robot id's to robots so they can be accessed easily
	std::map<int, Alice>::iterator map_it = alice_map.begin();
	//Creates an AliceBrain node, and subscribes/publishes to the necessary topics
	ros::init(argc, argv, "AliceBrain");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("alice_mail_array", 1000, aliceCallback);
	ros::Subscriber sub2 = n.subscribe("priority", 1000, prioritiesCallback);

	ros::Rate loopRate(50);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::robot_command_array > ("final_execute", 1000);
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::map > ("info_map", 1000);
	ros::Publisher pub3 = n.advertise < wvu_swarm_std_msgs::chargers > ("chargers", 1000);
	ros::Publisher pub4 = n.advertise < wvu_swarm_std_msgs::priorities > ("priority", 1000);
	ros::ServiceClient client = n.serviceClient < alice_swarm::get_maps > ("get_maps");

	while (ros::ok())
	{
		checkForUpdatedChargers(); //what it sounds like ###################

		// it!=mymap.end(); ++it)
		auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of Alice
		alice_swarm::get_maps srv;
		client.call(srv);

		std::vector<wvu_swarm_std_msgs::map> maps;
		for (int i = 0; i < srv.response.maps.size(); i++)
			maps.push_back(srv.response.maps.at(i)); //gets map info from srv

		std::vector<int> ids;
		for (int i = 0; i < srv.response.ids.size(); i++)
			ids.push_back(srv.response.ids.at(i)); //gets ids info from srv

		for (int i = 0; i < temp_mail.mails.size(); i++) //sends each bot its new info.
		{
			//==gives each robot the relative data it needs, whilst also creating the alice's
			alice_map[temp_mail.mails.at(i).name].updateModel(temp_mail.mails.at(i), maps, ids, &temp_mail.abs_chargers,
					temp_priorities.priorities.at(i).priority);
		}

		//=========Publish to info_map (pub2)==========
		map_it = alice_map.begin();
		while (map_it != alice_map.end())
		{
			map_it->second.model.pass(pub2);
			map_it++;
		}
		//======================================

		//=======================Publish final_execute (pub)=================
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
			//std::cout << temp.rid << " " << temp.r << " " << temp.theta << std::endl;

			execute.commands.push_back(temp); //adds vector to published vector
		}
		pub.publish(execute);
		//===============================================================

//	============Publish chargers (pub3)=============
		wvu_swarm_std_msgs::chargers chargers_to_publish;
		chargers_to_publish.charger = temp_mail.abs_chargers;
		if (temp_mail.abs_chargers.size() > 0) //publishes when there is actually stuff to publish.
		{
			prev_chargers.charger = temp_mail.abs_chargers;
			pub3.publish(chargers_to_publish);
		}
//	==========================================

//	============Publish Priorities (pub4)===========
		wvu_swarm_std_msgs::priorities priorities_to_publish;
//		std::cout<<"number of bot priors: "<<temp_priorities.priorities.size()<<std::endl;
		for (int i = 0; i < temp_priorities.priorities.size(); i++)
		{
			priorities_to_publish.priorities.push_back(temp_priorities.priorities.at(i));
		}
		if (temp_priorities.priorities.size() > 0) //publishes when there is actually stuff to publish.
		{
			pub4.publish(priorities_to_publish);
		}
//	==========================================

//		std::cout << "execute published" << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);
		std::cout << "Time taken by Alice: " << duration.count() << " microseconds" << std::endl;


		ros::spinOnce();
		loopRate.sleep();

	}
	return 0;
}
