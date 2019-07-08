#include "ros/ros.h"
#include <wvu_swarm_std_msgs/map.h>
#include <alice_swarm/get_map.h>
std::map<int, wvu_swarm_std_msgs::map> alice_map; //Maps robot id's to maps

void mapCallback(const wvu_swarm_std_msgs::map &msg)
{
	alice_map[msg.name]=msg;
}

bool getMap(alice_swarm::get_map::Request  &req,alice_swarm::get_map::Response &res)
{
  res.map = alice_map[req.name];
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_hub");
  ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("info_map", 1000, mapCallback);
  ros::ServiceServer service = n.advertiseService("get_map", getMap);
  ros::spin();
  return 0;
}
