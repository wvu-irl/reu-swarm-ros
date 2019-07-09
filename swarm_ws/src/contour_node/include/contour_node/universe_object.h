#ifndef UNIVERSE_H
#define UNIVERSE_H

#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <contour_node/level_description.h>

class Universe
{
private:
	wvu_swarm_std_msgs::map_levels overall_map;
public:
	Universe();
};

#endif
