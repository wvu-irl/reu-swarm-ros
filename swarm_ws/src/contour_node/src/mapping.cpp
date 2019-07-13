#include <ros/ros.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <contour_node/level_description.h>
#include <wvu_swarm_std_msgs/nuitrack_data.h>
#include <nuitrack_bridge/nuitrack_data.h>
#include <contour_node/gaussian_object.h>

#include <contour_node/universe_object.h>

#include <math.h>

// toggles verbose option
#define DEBUG 0

// toggles using testing equations
#define TEST_EQU 0

// toggles running NUI track test code
#define TEST_NUI 1

// toggles using a universe object to contain map data
#define RUN_UNIVERSE 1

// toggles if the main loop has a rate
#define RATE_LIMIT 1

#if DEBUG || TEST_EQU
#include <iostream>
#endif

// Global variables for nuitrack data
wvu_swarm_std_msgs::nuitrack_data g_nui;
geometry_msgs::Point leftProjected, rightProjected;
levelObject *g_selected = nullptr;

// Find x,y where the line passing between alpha and beta intercepts an xy plane at z
geometry_msgs::Point findZIntercept(geometry_msgs::Point _alpha,
		geometry_msgs::Point _beta, double _zed)
{
	/* THEORY
	 * Equation of line: r(t) = v*t+v0
	 * Direction vector: v = (xa - xb, ya - yb, za - zb)
	 * Offset vector: v0 = (xa, ya, za)
	 * Plug in z to r(t), solve for t, use t to solve for x and y/
	 */

	geometry_msgs::Point ret;

	// Check if no solution
	if (_alpha.z == _beta.z)
	{
		printf(
				"\033[1;31mhand_pointer: \033[0;31mNo solution for intercept\033[0m\n");
		ret.x = 0;
		ret.y = 0;
		ret.z = 0;
	}
	else
	{
		double t = (_zed - _alpha.z) / (_alpha.z - _beta.z);
		double x = _alpha.x * (t + 1) - _beta.x * t;
		double y = _alpha.y * (t + 1) - _beta.y * t;

		ret.x = x;
		ret.y = y;
		ret.z = _zed;
	}

	return ret;
}

#if RUN_UNIVERSE
static Universe universe; // creates a universe

// subscriber callback to add things to the universe
void additionCallback(wvu_swarm_std_msgs::obstacle obs)
{
	universe += obs;
}

void nuiCallback(wvu_swarm_std_msgs::nuitrack_data nui)
{
	// Copy the message
	g_nui = nui;

	// Find projections of hands onto table
	leftProjected = findZIntercept(g_nui.leftWrist, g_nui.leftHand, 0.0);
	rightProjected = findZIntercept(g_nui.rightWrist, g_nui.rightHand, 0.0);
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "mapping");
	ros::NodeHandle n;

	ros::Publisher map_pub = n.advertise < wvu_swarm_std_msgs::map_levels
			> ("/map_data", 1000);
	ros::Publisher left_pub = n.advertise < geometry_msgs::Point
			> ("/nui_bridge/hand_1", 1000);
	ros::Publisher right_pub = n.advertise < geometry_msgs::Point
			> ("/nui_bridge/hand_2", 1000);

	ros::Subscriber n_obs = n.subscribe("/add_obstacle", 1000,
			additionCallback);
	ros::Subscriber nuiSub = n.subscribe("/nuitrack_bridge", 1000, nuiCallback);

#if DEBUG
	std::cout << "Adding equation" << std::endl;
#endif

	levelObject *ptr;

	ptr = new gaussianObject(0, 0, "Gary", 10, 20, M_PI / 4.0, 10,
			map_ns::TARGET);
	universe += ptr;

	ptr = new gaussianObject(50, 0, "Larry", 5, 5, 0, 10, map_ns::TARGET);
	universe += ptr;

#if DEBUG
	std::cout << "\033[30;42mdone adding equation\033[0m" << std::endl;
#endif

#if RATE_LIMIT
	ros::Rate rate(60);
#endif

	geometry_msgs::Point *anchor = nullptr; // Where did the user's hand start when they grabbed a feature?

	while (ros::ok())
	{
		// If user's left hand (RIGHT) is open, points are free to move
		if (!g_nui.rightClick)
		{
			// Reset anchor since nothing is being manipulated
			anchor = nullptr;

			// Check if something is selected, if so lock point to it
			std::pair<double, double> leftProjPair(leftProjected.x, leftProjected.y);
                        g_selected = universe.findWithinRadius(leftProjPair, 10.0);
                        
			if (g_selected != nullptr)
			{
				leftProjected.x = g_selected->getOrigin().first;
				leftProjected.y = g_selected->getOrigin().second;
			}

			// If nothing is selected, move points freely
//                    if(leftProjected.x == 0.0 && leftProjected.y == 0.0 && leftProjected.z == 0.0)
			left_pub.publish(leftProjected);
//                    if(rightProjected.x == 0.0 && rightProjected.y == 0.0 && rightProjected.z == 0.0)
			right_pub.publish(rightProjected);
		}
		// If user's left hand (RIGHT) is closed, maybe modify objects
		else
		{
			ROS_INFO("Hand closed!");
			// If nothing had been selected before hand was closed, do nothing
			if (g_selected != nullptr)
			{
				ROS_INFO("%s", g_selected->getName().c_str());
				// If object was just grabbed, set the anchor
				if (anchor == nullptr)
				{
					anchor = new geometry_msgs::Point;
					anchor->x = g_nui.leftHand.x;
					anchor->y = g_nui.leftHand.y;
					anchor->z = g_nui.leftHand.z;
				}

				// Manipulate the object
				g_selected->nuiManipulate(g_nui.leftHand.x - anchor->x,
						g_nui.leftHand.y - anchor->y, g_nui.leftHand.z - anchor->z);

				// Update anchor so it's one iteration behind hand
				anchor->x = g_nui.leftHand.x;
				anchor->y = g_nui.leftHand.y;
				anchor->z = g_nui.leftHand.z;

				ROS_INFO("Anchor moved!");
			}
		}

		map_pub.publish(universe.getPublishable());

		ros::spinOnce();
#if RATE_LIMIT
		rate.sleep();
#endif
	}
}
