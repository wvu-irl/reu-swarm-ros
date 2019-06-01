#include <ros/ros.h>



int main(int argc, char **argv)
{
	ros::init(argc, argv, "ros_server_tester");
	ros::NodeHandle n;

	ros::Publisher send = n.advertise<server_setup::robotcommand>("execute", 1000);
}
