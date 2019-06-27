#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/flows.h>

#define VOBJ_DEBUG 0

wvu_swarm_std_msgs::vicon_points temp_targets;
wvu_swarm_std_msgs::vicon_points temp_temp_targets;
wvu_swarm_std_msgs::vicon_points temp_obs;
//wvu_swarm_std_msgs::vicon_points temp_bots;
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
#if VOBJ_DEBUG
    auto dist = [](wvu_swarm_std_msgs::vicon_point a, wvu_swarm_std_msgs::vicon_point b)->double{
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    };
    
    std::cout << "Setting points to: " << std::endl;
    for (size_t i = 0;i < msg.point.size();i++)
    {
        if (i < temp_targets.point.size())
            std::cout << "\t[" << i << "]:" << (dist(temp_targets.point[i], msg.point[i]) < 3 ? "\033[32m" : "\033[30;41m") << "(" 
                    << msg.point[i].x << ", " << msg.point[i].y << ")\033[0m <-- (" << temp_targets.point[i].x << ", " 
                    << temp_targets.point[i].y << ")" << std::endl;
        else
            std::cout << "\t[" << i << "]: (" << msg.point[i].x << ", " << msg.point[i].y << ")" << std::endl;
    }
#endif
        temp_targets = msg; // resets current target point
}

void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
	temp_obs = msg; // resets current obstacles' points
}

/**
 * 
 * creates a flow for hitting a puck to a specified location
 * 
 * @param in_vector where to store the flow vectors
 * @param target    location of target point
 * @param to        end point
 * @param sid       id for simulated object
 */
void createPuckFlow(wvu_swarm_std_msgs::flows &in_vector, wvu_swarm_std_msgs::vicon_point target,
		std::pair<float, float> to, int sid)
{
        // expanding input
	float x = target.x; // target point
	float y = target.y;
	float r = pow(pow(to.second - y, 2) + pow(to.first - x, 2), 0.5); // distance to destination
	float xd = to.first - x; // velocity vector
	float yd = to.second - y;

//		//top
	wvu_swarm_std_msgs::flow cur0; // creating a local temporary flow vecotr
	cur0.x = x + (xd) / r * 10; // calculating direction
	cur0.y = y + (yd) / r * 10;
	cur0.r = 1;
	cur0.pri = 1;
	cur0.theta = atan2(yd, xd);
	cur0.sid = sid;
	in_vector.flow.push_back(cur0); // adding to list
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
	cur4.r = 1;
	in_vector.flow.push_back(cur4);

	//bottomright
	wvu_swarm_std_msgs::flow cur5(cur1);
	cur5.x -= xd / r * 20;
	cur5.y -= yd / r * 20;
	cur5.theta = cur1.theta - 5 * M_PI / 6;
	cur5.r = 1;
	in_vector.flow.push_back(cur5);

	//middle
	wvu_swarm_std_msgs::flow cur6;
	cur6.x = x;
	cur6.y = y;
	cur6.r = 1;
	cur6.theta = atan2(yd, xd);
	cur6.sid = sid;
	cur6.pri= 1;
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
        // end of adding things to list
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
	wvu_swarm_std_msgs::vicon_point cur0; // creates point

        // sets point values at start
	cur0.x = 0.01;
	cur0.y = 0;
	cur0.sid = 0;

	in_vector.point.push_back(cur0);  // adding to point list
        
#if VOBJ_DEBUG
        std::cout << "Num targets: " << (in_vector.point.size() > 1 ? "\033[41;30m" : "\033[32m") << in_vector.point.size() << "\033[0m" << std::endl;
#endif
        
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

/**
 * publishes all the obstacles that are in the system
 * 
 * @param _pub publisher that publishes to an obstacles topic
 */
void makeObstacles(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::vicon_points vp_vector; // points
	createBoundary(vp_vector); // creates bounding box for table
	_pub.publish(vp_vector); // publishes points

}

/**
 * publishes all the targets/goals/food in the system
 * 
 * @param _pub target publisher
 */
void makeTargets(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::vicon_points vp_vector; // points container
	createFood(vp_vector); // creates a single target
	_pub.publish(vp_vector); // publishes puck

}

/**
 * Publishes all the flows in the system
 * 
 * Flows are paths that designate how a robot should move locally
 * 
 * @param _pub flow publisher
 */
void makeFlows(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::flows vp_vector; // points
	std::pair<float, float> goal1(0,100); // creating goal 1
	std::pair<float,float> goal2(0,-100); // creating goal 2
	if (temp_targets.point.size() != 0) { // checking that there are targets to form flows from
		createPuckFlow(vp_vector,temp_targets.point.at(0),goal1,1); // creates flow around puck
		createPuckFlow(vp_vector,temp_targets.point.at(0),goal2,2);

	}

	_pub.publish(vp_vector); // publishing
}

int main(int argc, char **argv) // begin here
{
        // ros initialize
	ros::init(argc, argv, "virtual_objects");
	ros::NodeHandle n;
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_obstacles", 1000); // pub to obstacles
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_targets", 1000); // pub to targets
	ros::Publisher pub3 = n.advertise < wvu_swarm_std_msgs::flows > ("virtual_flows", 1000); // pub to flows

        // subscribing
	ros::Subscriber sub1 = n.subscribe("virtual_targets", 1000, pointCallback);
	ros::Subscriber sub2 = n.subscribe("virtual_obstacles", 1000, obsCallback);
	//ros::Subscriber	sub2 = n.subscribe("vicon_array",1000,botCallback;
	ros::Rate loopRate(100);
	sleep(2); //waits for sim to be awake
	int i = 0;
	while (ros::ok() && i < 10000) // setup loop
	{
		makeObstacles(pub1); // creating obstacles
		makeTargets(pub2); // creating targets
		makeFlows(pub3); // creating flows
		ros::spinOnce(); // spinning callbacks
//		loopRate.sleep();
                usleep(10);
		i++; // incrementing counter

	}
#if VOBJ_DEBUG
       std::cout << "\033[35;1mStarting second loop\033[0m" << std::endl; 
#endif 
	while (ros::ok()) // main loop
	{
		makeObstacles(pub1); // publishing obstacles
		makeFlows(pub3); // publishing flows
		ros::spinOnce(); // spinning callbacks
		loopRate.sleep();

	}
}
