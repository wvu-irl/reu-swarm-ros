#include <ros/ros.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <contour_node/level_description.h>
#include <wvu_swarm_std_msgs/nuitrack_data.h>
#include <nuitrack_bridge/nuitrack_data.h>

#include <math.h>

#define DEBUG 0
#define TEST_EQU 1

#if DEBUG
#include <iostream>
#endif

// Global variables for nuitrack data
wvu_swarm_std_msgs::nuitrack_data g_nui;
geometry_msgs::Point leftProjected, rightProjected;

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
		ret.x = 0.0;
		ret.y = 0.0;
		ret.z = 0.0;
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

static wvu_swarm_std_msgs::map_levels overall_map;

void newObs(wvu_swarm_std_msgs::obstacle obs)
{
	while (overall_map.levels.size() <= obs.level)
	{
#if DEBUG
		std::cout << "\033[30;43mAdded level: " << overall_map.levels.size()
				<< "\033[0m" << std::endl;
#endif
		overall_map.levels.push_back(wvu_swarm_std_msgs::map_level());
	}
	bool found = false;

	for (size_t i = 0;
			i < overall_map.levels.at(obs.level).functions.size() && !found; i++)
	{
		if (overall_map.levels.at(obs.level).functions.at(i).name.compare(
				obs.characteristic.name) == 0)
		{
			overall_map.levels.at(obs.level).functions[i] = obs.characteristic;
			found = true;
		}
	}

	if (!found)
		overall_map.levels[obs.level].functions.push_back(obs.characteristic);

	if (obs.level != map_ns::COMBINED)
	{
		wvu_swarm_std_msgs::obstacle comb;
		obs.characteristic.amplitude *= 1 - ((obs.level % 2) * 2);
		comb.characteristic = obs.characteristic;
		comb.level = map_ns::COMBINED;

		newObs(comb);
	}
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
	ros::Subscriber n_obs = n.subscribe("/add_obstacle", 1000, newObs);
	ros::Subscriber nuiSub = n.subscribe("/nuitrack_bridge", 1000, nuiCallback);

	///////////////////////////////////////////////////////////////////
	// default map setup
#if DEBUG
	std::cout << "Adding equation" << std::endl;
#endif

#if TEST_EQU
	wvu_swarm_std_msgs::ellipse el;
	el.x_rad = 5;
	el.y_rad = 2;
	el.theta_offset = M_PI_4;

	wvu_swarm_std_msgs::gaussian gaus;
	gaus.ellipse = el;
	gaus.ellipse.offset_x = 0;
	gaus.ellipse.offset_y = 0;
	gaus.amplitude = 20;
	gaus.name = "Bob";

	wvu_swarm_std_msgs::obstacle obs;
	obs.characteristic = gaus;
	obs.level = map_ns::TARGET;

	newObs(obs);

	el.x_rad = 4;
	el.y_rad = 7;
	el.theta_offset = 0;

	gaus.ellipse = el;
	gaus.ellipse.offset_x = 10;
	gaus.ellipse.offset_y = 0;
	gaus.amplitude = 10;
	gaus.name = "Jeff";

	obs.characteristic = gaus;
	obs.level = map_ns::TARGET;

	newObs(obs);
#endif
#if DEBUG
	std::cout << "\033[30;42mdone adding equation\033[0m" << std::endl;
#endif
	// end default map setup
	///////////////////////////////////////////////////////////////////

	ros::Rate rate(60);

#if TEST_EQU
	int tick = 0;
#endif

	while (ros::ok())
	{
#if TEST_EQU
		tick++;
		tick %= 1000;

		el.x_rad = 5;
		el.y_rad = 2;
		el.theta_offset = tick * M_PI / 100;

		gaus.ellipse = el;
		gaus.ellipse.offset_x = 0;
		gaus.ellipse.offset_y = 0;
		gaus.amplitude = 20;
		gaus.name = "Bob";

		obs.characteristic = gaus;
		obs.level = map_ns::TARGET;
		newObs(obs);
#endif

		map_pub.publish(overall_map);

		ros::spinOnce();
		rate.sleep();
	}
}
