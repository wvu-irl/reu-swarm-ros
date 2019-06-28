#include <ros/ros.h>
#include <contour_node/obstacle.h>
#include <contour_node/map_levels.h>
#include <contour_node/level_description.h>

static contour_node::map_levels overall_map;

void newObs(contour_node::obstacle obs)
{
	while (overall_map.levels.size() <= obs.level)
	{
		overall_map.levels.push_back(contour_node::map_level());
	}
	overall_map.levels[obs.level].functions.push_back(obs.characteristic);

	if (obs.level != map_ns::COMBINED)
	{
		contour_node::obstacle comb;
		comb.characteristic = obs.characteristic;
		comb.level = map_ns::COMBINED;

		newObs(comb);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "mapping");
	ros::NodeHandle n;

	ros::Publisher map_pub = n.advertise < contour_node::map_levels
			> ("/map_data", 1000);
	ros::Subscriber n_obs = n.subscribe("/add_obstacle", 1000, newObs);

	ros::Rate rate(60);

	while (ros::ok())
	{
		map_pub.publish(overall_map);

		ros::spinOnce();
		rate.sleep();
	}
}
