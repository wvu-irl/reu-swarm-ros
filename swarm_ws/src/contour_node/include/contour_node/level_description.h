#ifndef LEVEL_DESCRIPTION_H
#define LEVEL_DESCRIPTION_H

#include <wvu_swarm_std_msgs/vicon_point.h>
#include <contour_node/map_levels.h>


namespace map
{


// current levels that are included in the overall map
enum LEVEL
{
	TARGET,
	OBJECT
};

/**
 * Calculates the z value at a given point for a given map
 *
 * @param ml is a map level that you are calculating a point on
 * @param loc is the current location of the robot
 *
 */
double calculate(contour_node::map_level ml, wvu_swarm_std_msgs::vicon_point loc);

};
#endif
