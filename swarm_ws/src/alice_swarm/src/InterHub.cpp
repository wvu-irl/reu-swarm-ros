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
