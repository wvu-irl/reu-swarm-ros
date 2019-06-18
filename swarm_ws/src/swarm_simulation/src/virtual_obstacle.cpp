#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>

#include <wvu_swarm_std_msgs/vicon_points.h>

void createBoundary(wvu_swarm_std_msgs::vicon_points &in_vector)
{
	for (int i = 0; i < 20; i++)
	{

		wvu_swarm_std_msgs::vicon_point cur0;
		cur0.x = -50;
		cur0.y = -100 + 10 * i;
		wvu_swarm_std_msgs::vicon_point cur1;
		cur1.x = 50;
		cur1.y = 100 - 10 * i;
		in_vector.point.push_back(cur0);
		in_vector.point.push_back(cur1);
	}
	for (int i = 0; i < 10; i++)
	{
		wvu_swarm_std_msgs::vicon_point cur0;
		cur0.x = 50 - 10 * i;
		cur0.y = -100;
		wvu_swarm_std_msgs::vicon_point cur1;
		cur1.x = -50 + 10 * i;
		cur1.y = 100;
		in_vector.point.push_back(cur0);
		in_vector.point.push_back(cur1);
	}

	wvu_swarm_std_msgs::vicon_point cur;
	cur.x=0;
	cur.y=0;
	in_vector.point.push_back(cur);

}

void makeAndPublish(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::vicon_points vp_vector;
	createBoundary(vp_vector);
	_pub.publish(vp_vector);

}
int main(int argc, char **argv)
{
	ros::init(argc, argv, "virtual_obstacle");
	ros::NodeHandle n;
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_obstacles", 1000);
	ros::Rate loopRate(10);
	while (ros::ok())
	{
		makeAndPublish(pub);
		ros::spinOnce();
		loopRate.sleep();
	}
}
