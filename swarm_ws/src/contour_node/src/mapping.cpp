#include <ros/ros.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <contour_node/level_description.h>

#include <math.h>

#define DEBUG 0
#define TEST_EQU 1

#if DEBUG
#include <iostream>
#endif

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

	for (size_t i = 0; i < overall_map.levels.at(obs.level).functions.size() && !found; i++)
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

int main(int argc, char **argv)
{
	ros::init(argc, argv, "mapping");
	ros::NodeHandle n;

	ros::Publisher map_pub = n.advertise < wvu_swarm_std_msgs::map_levels
			> ("/map_data", 1000);
	ros::Subscriber n_obs = n.subscribe("/add_obstacle", 1000, newObs);

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

	el.x_rad = 7;
	el.y_rad = 4;
	el.theta_offset = 0;

	gaus.ellipse = el;
	gaus.ellipse.offset_x = 5;
	gaus.ellipse.offset_y = 0;
	gaus.amplitude = 10;
	gaus.name = "Jeff";

	obs.characteristic = gaus;
	obs.level = map_ns::OBSTACLE;

	newObs(obs);
#endif
#if DEBUG
	std::cout << "\033[30;42mdone adding equation\033[0m" << std::endl;
#endif
	// end default map setup
	///////////////////////////////////////////////////////////////////

	ros::Rate rate(60);

	while (ros::ok())
	{
		map_pub.publish(overall_map);

		ros::spinOnce();
		rate.sleep();
	}
}
