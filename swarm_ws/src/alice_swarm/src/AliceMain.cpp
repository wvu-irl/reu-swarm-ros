/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

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

#define DEBUG_chargers 1

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
#if DEBUG_chargers
		std::cout<<"+++++++++++++++Compare++++++++++++++++++++"<<std::endl;
#endif

		while(i < prev_chargers.charger.size() && same)
		{
#if DEBUG_chargers
			std::cout<<"charger: "<<i<<"|prev charger:"<<(prev_chargers.charger.at(i).occupied? "true" : "false")<<std::endl;
			std::cout<<"charger: "<<i<<"|new charger:"<<(temp_mail.abs_chargers.at(i).occupied? "true" : "false")<<std::endl;
#endif

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
		wvu_swarm_std_msgs::robot_command_array execute;
		for (int i = 0; i < temp_mail.mails.size(); i++) //sends each bot its new info.
		{
			//==gives each robot the relative data it needs, whilst also creating the alice's
			alice_map[temp_mail.mails.at(i).name].updateModel(temp_mail.mails.at(i), maps, ids, &temp_mail.abs_chargers,
					temp_priorities.priorities.at(i).priority);
			wvu_swarm_std_msgs::robot_command temp;
						AliceStructs::vel tempVel = alice_map[temp_mail.mails.at(i).name].generateVel();
						temp.rid = temp_mail.mails.at(i).name;

						if (tempVel.mag > 1) //caps the speed
							temp.r = 1;
						else
							temp.r = tempVel.mag;
						//std::cout << tempVel.dir << std::endl;
						temp.theta = 180 / M_PI * fmod(2 * M_PI + tempVel.dir, 2 * M_PI);
						//std::cout << temp.rid << " " << temp.r << " " << temp.theta << std::endl;

						execute.commands.push_back(temp); //adds vector to published vector
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
		pub.publish(execute);
		//===============================================================

		//============Publish chargers (pub3)=============
		wvu_swarm_std_msgs::chargers chargers_to_publish;
		chargers_to_publish.charger = temp_mail.abs_chargers;
		if (temp_mail.abs_chargers.size() > 0) //publishes when there is actually stuff to publish.
		{
#if DEBUG_chargers
			std::cout<<"****************Published*********************"<<std::endl;
			for(int i = 0; i < temp_mail.abs_chargers.size(); i ++)
			{
				std::cout<<"charger: "<<i<<" |"<<(temp_mail.abs_chargers.at(i).occupied? "true" : "false")<<std::endl;
			}
			std::cout<<"*********************************************"<<std::endl;
#endif
			prev_chargers.charger = temp_mail.abs_chargers;
			pub3.publish(chargers_to_publish);
		}
		//==========================================

		//============Publish Priorities (pub4)===========
		wvu_swarm_std_msgs::priorities priorities_to_publish;
		//std::cout<<"number of bot priors: "<<temp_priorities.priorities.size()<<std::endl;
		for (int i = 0; i < temp_priorities.priorities.size(); i++)
		{
			priorities_to_publish.priorities.push_back(temp_priorities.priorities.at(i));
		}
		if (temp_priorities.priorities.size() > 0) //publishes when there is actually stuff to publish.
		{
			pub4.publish(priorities_to_publish);
		}
		//==========================================

		//std::cout << "execute published" << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);

		//std::cout << "Time taken by Alice: " << duration.count() << " microseconds" << std::endl;
		//is_updated = true;
		ros::spinOnce();
		loopRate.sleep();

	}
	return 0;
}
