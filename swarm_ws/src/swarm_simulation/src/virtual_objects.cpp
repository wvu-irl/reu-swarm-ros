#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <math.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/flows.h>
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

//	for (int i=0; i<20; i++)
//	{
//	wvu_swarm_std_msgs::vicon_point cur0;
//	cur0.x=25*cos(2*M_PI*i/20);
//	cur0.y=50*sin(2*M_PI*i/20);
//	in_vector.point.push_back(cur0);
//	wvu_swarm_std_msgs::vicon_point cur1;
//		cur1.x=15*cos(2*M_PI*i/20);
//		cur1.y=30*sin(2*M_PI*i/20);
//		in_vector.point.push_back(cur1);
//		wvu_swarm_std_msgs::vicon_point cur2;
//			cur2.x=5*cos(2*M_PI*i/20);
//			cur2.y=10*sin(2*M_PI*i/20);
//			in_vector.point.push_back(cur2);
//	}

}

void createFood(wvu_swarm_std_msgs::vicon_points &in_vector)
{
	wvu_swarm_std_msgs::vicon_point cur0;
	cur0.x = 0;
	cur0.y = -80;
	in_vector.point.push_back(cur0);
//	for (int i=0; i<20; i++)
//	{
//	wvu_swarm_std_msgs::vicon_point cur0;
//	cur0.x=25*cos(2*M_PI*i/20);
//	cur0.y=50*sin(2*M_PI*i/20);
//	in_vector.point.push_back(cur0);
//	wvu_swarm_std_msgs::vicon_point cur1;
//		cur1.x=15*cos(2*M_PI*i/20);
//		cur1.y=30*sin(2*M_PI*i/20);
//		in_vector.point.push_back(cur1);
//		wvu_swarm_std_msgs::vicon_point cur2;
//			cur2.x=5*cos(2*M_PI*i/20);
//			cur2.y=10*sin(2*M_PI*i/20);
//			in_vector.point.push_back(cur2);
//	}

}

void makeObstacles(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::vicon_points vp_vector;
	createBoundary(vp_vector);
	_pub.publish(vp_vector);

}

void makeTargets(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::vicon_points vp_vector;
	//createFood(vp_vector);
	_pub.publish(vp_vector);

}

void makeFlows(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::flows vp_vector;
	wvu_swarm_std_msgs::flow cur0;
	cur0.x = 0;
	cur0.y = 0;
	cur0.r=100;
	cur0.theta=0;
	vp_vector.flow.push_back(cur0);
	_pub.publish(vp_vector);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "virtual_objects");
	ros::NodeHandle n;
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_obstacles", 1000);
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_targets", 1000);
	ros::Publisher pub3 = n.advertise < wvu_swarm_std_msgs::flows > ("virtual_flows", 1000);
	ros::Rate loopRate(50);
	while (ros::ok())
	{
		makeObstacles(pub1);

		makeFlows(pub3);
		ros::spinOnce();
		loopRate.sleep();
<<<<<<< HEAD
		i++;

	}
	while (ros::ok())
	{
		makeObstacles(pub1);
		makeTargets(pub2);
		makeFlows(pub3);
		ros::spinOnce();
		loopRate.sleep();

=======
>>>>>>> refs/remotes/origin/master
	}
}
