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
#include <wvu_swarm_std_msgs/map.h>
#include <alice_swarm/get_maps.h>
#include <alice_swarm/get_map.h>

std::map<int, wvu_swarm_std_msgs::map> alice_map; //Maps robot id's to maps

void mapCallback(const wvu_swarm_std_msgs::map &msg)
{
	alice_map[msg.name]=msg;
}
//void pathCallback();

bool getMap(alice_swarm::get_map::Request  &req,alice_swarm::get_map::Response &res)
{
  res.map = alice_map[req.name];
  return true;
}

bool getMaps(alice_swarm::get_maps::Request  &req,alice_swarm::get_maps::Response &res)
{
	res.ids.clear();
	res.maps.clear();
	std::map<int,wvu_swarm_std_msgs::map>::iterator it = alice_map.begin();
	while (it != alice_map.end()){
		res.ids.push_back(it->first);
	  res.maps.push_back(it->second);
		it++;
	}
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "inter_hub");
  ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("info_map", 1000, mapCallback);
	//ros::Subscriber sub2 = n.subscribe("ideal_paths",1000,pathCallback;
  ros::ServiceServer service = n.advertiseService("get_map", getMap);
  ros::ServiceServer service2 = n.advertiseService("get_maps", getMaps);
  ros::spin();
  return 0;
}
