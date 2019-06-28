#include <iostream>
#include <visualization/alice_pov.h>
#include <ros/ros.h>

int main(int argc, char **argv)
{
	AlicePOV pov;
	ros::init(argc, argv, "map");
	ros::NodeHandle n;
	pov.Run(n);
	return 0;
}
