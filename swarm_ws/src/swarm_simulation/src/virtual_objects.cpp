#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <math.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/flows.h>

wvu_swarm_std_msgs::vicon_points temp_targets;
//wvu_swarm_std_msgs::vicon_points temp_bots;
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{

	temp_targets = msg;
}

//creates a flow for hitting a puck to a specified location
void createPuckFlow(wvu_swarm_std_msgs::flows &in_vector, wvu_swarm_std_msgs::vicon_point target, std::pair<float, float> to, int sid)
{

	float x = target.x;
	float y = target.y;
	float r = pow(pow(to.second - y, 2) + pow(to.first-x, 2), 0.5);
	float xd=to.first-x;
	float yd=to.second-y;

//		//top
	wvu_swarm_std_msgs::flow cur0;
	cur0.x = x +(xd) / r * 10;
	cur0.y = y + (yd) / r * 10;
	cur0.r = 10;
	cur0.theta = atan2(yd,xd);
	cur0.sid = sid;
	in_vector.flow.push_back(cur0);
//
//		//topright
	wvu_swarm_std_msgs::flow cur1(cur0);
	cur1.x += (yd) / r * 10;
	cur1.y -= (xd) / r * 10;
	cur1.theta = cur0.theta - M_PI_2;
	in_vector.flow.push_back(cur1);

	//topleft
	wvu_swarm_std_msgs::flow cur2(cur0);
	cur2.x -= yd / r * 10;
	cur2.y += xd / r * 10;
	cur2.theta = cur0.theta + M_PI_2;
	in_vector.flow.push_back(cur2);
//
//		//bottom
	wvu_swarm_std_msgs::flow cur3(cur0);
	cur3.x -= xd / r * 20;
	cur3.y -= yd / r * 20;
	cur3.theta = cur0.theta;
	in_vector.flow.push_back(cur3);

	//bottomleft
	wvu_swarm_std_msgs::flow cur4(cur2);
	cur4.x -= xd / r * 20;
	cur4.y -= yd / r * 20;
	cur4.theta = cur2.theta + 5 * M_PI / 6;
	cur4.r = 10;
	in_vector.flow.push_back(cur4);

	//bottomright
	wvu_swarm_std_msgs::flow cur5(cur1);
	cur5.x -= xd / r * 20;
	cur5.y -= yd / r * 20;
	cur5.theta = cur1.theta - 5 * M_PI / 6;
	cur5.r = 10;
	in_vector.flow.push_back(cur5);

	//middle
	wvu_swarm_std_msgs::flow cur6;
	cur6.x = x;
	cur6.y = y;
	cur6.r = 10;
	cur6.theta = atan2(yd,xd);
	cur6.sid = sid;
	in_vector.flow.push_back(cur6);

	//right
	wvu_swarm_std_msgs::flow cur7(cur6);
	cur7.x += yd / r * 20;
	cur7.y -= xd / r * 20;
	cur7.theta = cur6.theta + M_PI;
	in_vector.flow.push_back(cur7);

	//left
	wvu_swarm_std_msgs::flow cur8(cur6);
	cur8.x -= yd / r * 20;
	cur8.y += xd / r * 20;
	cur8.theta = cur6.theta + M_PI;
	in_vector.flow.push_back(cur8);

	//left2
	wvu_swarm_std_msgs::flow cur9(cur6);
	cur9.x -= yd / r * 10;
	cur9.y += xd / r * 10;
	cur9.theta = cur6.theta + 3 * M_PI / 4;
	in_vector.flow.push_back(cur9);

	//right2
	wvu_swarm_std_msgs::flow cur10(cur6);
	cur10.x += yd / r * 10;
	cur10.y -= xd / r * 10;
	cur10.theta = cur6.theta - 3 * M_PI / 4;
	in_vector.flow.push_back(cur10);
}

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

}

void createFood(wvu_swarm_std_msgs::vicon_points &in_vector)
{
	//is the puck
	wvu_swarm_std_msgs::vicon_point cur0;
	cur0.x = 1;
	cur0.y = 1;
	cur0.sid = 0;
	in_vector.point.push_back(cur0);
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
	createFood(vp_vector);
	_pub.publish(vp_vector);

}

void makeFlows(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::flows vp_vector;
	std::pair<float, float> goal1(0,100);
	std::pair<float,float> goal2(0,-100);
	if (temp_targets.point.size() != 0) {
		createPuckFlow(vp_vector,temp_targets.point.at(0),goal1,1);
		createPuckFlow(vp_vector,temp_targets.point.at(0),goal2,2);
	}

	_pub.publish(vp_vector);

}

//whirlpool
//for (int i = 0; i < 20; i++)
//{
//
//	wvu_swarm_std_msgs::flow cur0;
//	cur0.x = 25 * cos(2 * M_PI * i / 20);
//	cur0.y = 50 * sin(2 * M_PI * i / 20);
//	cur0.r = 10;
//	cur0.theta = 2 * M_PI * i / 20 + M_PI_2;
//	vp_vector.flow.push_back(cur0);
//	wvu_swarm_std_msgs::flow cur1;
//	cur1.x = 15 * cos(2 * M_PI * i / 20);
//	cur1.y = 30 * sin(2 * M_PI * i / 20);
//	cur1.r = 10;
//	cur1.theta = 2 * M_PI * i / 20 + M_PI_2;
//	vp_vector.flow.push_back(cur1);
//}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "virtual_objects");
	ros::NodeHandle n;
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_obstacles", 1000);
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_targets", 1000);
	ros::Publisher pub3 = n.advertise < wvu_swarm_std_msgs::flows > ("virtual_flows", 1000);
	ros::Subscriber sub1 = n.subscribe("virtual_targets", 1000, pointCallback);
	//ros::Subscriber	sub2 = n.subscribe("vicon_array",1000,botCallback;
			ros::Rate loopRate(50);
			sleep(1); //waits for sim to be awake
			int i = 0;
			while (ros::ok() && i < 50)
			{
				makeObstacles(pub1);
				makeTargets(pub2);
				makeFlows(pub3);
				ros::spinOnce();
				loopRate.sleep();
				i++;

			}
			while (ros::ok())
			{
				makeObstacles(pub1);

				makeFlows(pub3);
				ros::spinOnce();
				loopRate.sleep();

			}
		}
