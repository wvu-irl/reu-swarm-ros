#include <ros/ros.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <wvu_swarm_std_msgs/nuitrack_data.h>
#include <contour_node/level_description.h>
#include <contour_node/gaussian_object.h>
#include <contour_node/universe_object.h>
#include <nuitrack_bridge/nuitrack_data.h>
#include <std_msgs/String.h>

#include <chrono>
#include <math.h>
#include <string>

// toggles verbose option
#define DEBUG 0

// toggles if the main loop has a rate
#define RATE_LIMIT 1

#if DEBUG
#include <iostream>
#endif

// Global variables for nuitrack data
wvu_swarm_std_msgs::nuitrack_data g_nui; // State of the user
geometry_msgs::Point leftProjected, rightProjected; // Point for hand on table
levelObject *g_selected = nullptr; // Object currently selected by user
std::pair<double, double> originOffset(0.0, 0.0); // Move the nuitrack frame if needed
const double boundsX = 25.0, boundsY = 50.0; // Bounds before the origin will start moving
const double originShift = 0.02; // Rate to move the origin towards user's hand if out of bounds

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
static Universe universe; // creates a universe
// subscriber callback to add things to the universe
void additionCallback(wvu_swarm_std_msgs::obstacle obs)
{
#if DEBUG
	std::cout << "Adding: " << obs.characteristic << " , " << obs.level << std::endl;
#endif
	universe += obs;
}

void removeCallback(std_msgs::String str)
{
	universe -= std::string(str.data);
}

void nuiCallback(wvu_swarm_std_msgs::nuitrack_data nui)
{
	// Copy the message
	g_nui = nui;

	// Find projections of hands onto table
	leftProjected = findZIntercept(g_nui.leftWrist, g_nui.leftHand, 0.0);
	leftProjected.x += originOffset.first;
	leftProjected.y += originOffset.second;
	rightProjected = findZIntercept(g_nui.rightWrist, g_nui.rightHand, 0.0);
	rightProjected.x += originOffset.first;
	rightProjected.y += originOffset.second;

#if DEBUG
        std::cout << "lx = " << leftProjected.x << "\tly = " << leftProjected.y << std::endl;
#endif

	// Check if left hand is out of bounds only if hand is open
	if (!nui.rightClick)
	{
		// If too far in either axis, shift the origin offset towards point
		if (leftProjected.x < -boundsX)
			originOffset.first += originShift * (boundsX - leftProjected.x);
		else if (leftProjected.x > boundsX)
			originOffset.first -= originShift * (leftProjected.x - boundsX);
		if (leftProjected.y < -boundsY)
			originOffset.second += originShift * (boundsX - leftProjected.y);
		else if (leftProjected.y > boundsY)
			originOffset.second -= originShift * (leftProjected.y - boundsY);
#if DEBUG
            std::cout << "ox = " << originOffset.first << "\toy = " << originOffset.second << std::endl;
#endif
		// Don't bother applying shift yet, that will occur next iteration
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "mapping");
	ros::NodeHandle n;

	ros::NodeHandle n_priv("~");

	bool use_keyboard;
	n_priv.param<bool>("use_keyboard", use_keyboard, false);

	ros::Publisher map_pub = n.advertise < wvu_swarm_std_msgs::map_levels
			> ("/map_data", 1000);

	ros::Publisher left_pub = n.advertise < geometry_msgs::Point
			> ("hand_1", 1000);
	ros::Publisher right_pub = n.advertise < geometry_msgs::Point
			> ("hand_2", 1000);
	if (!use_keyboard)
	{
		ros::Subscriber nuiSub = n.subscribe("/nuitrack_bridge/rolling_average",
				1000, nuiCallback);
	}
	ros::Subscriber n_obs = n.subscribe("/add_obstacle", 1000, additionCallback);
	ros::Subscriber r_obs = n.subscribe("/rem_obstacle", 1000, removeCallback);
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
	bool prevGestureFound = false;
	while (ros::ok())
	{
		if (!use_keyboard)
		{
			// If user's left hand (RIGHT) is open, points are free to move
			if (!g_nui.rightClick)
			{
				// Reset anchor since nothing is being manipulated
				anchor = nullptr;

				// Check if something should be selected.
				//   This will return a pointer to a modifiable COPY of the
				//   object selected. Be sure to add it back into universe
				//   after altering.
				std::pair<double, double> leftProjPair(leftProjected.x,
						leftProjected.y);
				g_selected = universe.findWithinRadius(leftProjPair, 10.0);

				// If something is selected, lock hand to it
				if (g_selected != nullptr)
				{
					leftProjected.x = g_selected->getOrigin().first;
					leftProjected.y = g_selected->getOrigin().second;
				}
				// If nothing is selected, check for gestures
				else
				{
					std::cout << (int) g_nui.gestureData << std::endl;

					// Add a new feature if PUSH is detected
					if (g_nui.gestureData == (char) gestureType::PUSH
							&& !prevGestureFound)
					{
						std::cout << "ADDING GAUSSIAN" << std::endl;
						prevGestureFound = true; // Change flag to prevent duplicates

						std::string gausName = "nui_"
								+ std::to_string(ros::Time::now().sec);

						// Construct new object at right (left) hand
						ptr = new gaussianObject(leftProjected.x, leftProjected.y, gausName,
								10, 10, 0, 10, map_ns::TARGET);
						universe += ptr; // Add to universe

						std::cout << universe << std::endl;
					}
					else if (g_nui.gestureData != (char) gestureType::PUSH)
					{
						prevGestureFound = false;
					}
				}

				left_pub.publish(leftProjected);
				right_pub.publish(rightProjected);
			}
			// If user's left hand (RIGHT) is closed, maybe modify objects
			else
			{
				ROS_INFO("Hand closed!");
				// If nothing had been selected before hand was closed, do nothing
				// If something was selected, move it
				if (g_selected != nullptr)
				{
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
					universe += g_selected; // Update changes in universe

					ROS_INFO("Moved %s by %03.1f, %03.1f, %03.1f!",
							g_selected->getName().c_str(), g_nui.leftHand.x - anchor->x,
							g_nui.leftHand.y - anchor->y, g_nui.leftHand.z - anchor->z);

					// Update anchor so it's one iteration behind hand
					anchor->x = g_nui.leftHand.x;
					anchor->y = g_nui.leftHand.y;
					anchor->z = g_nui.leftHand.z;
				}
			}
		}
		else // using keyboard
		{
			// TODO implement this
		}
		map_pub.publish(universe.getPublishable());

		ros::spinOnce();
#if RATE_LIMIT
		rate.sleep();
#endif
	}
}
