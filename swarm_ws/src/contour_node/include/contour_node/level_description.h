#ifndef LEVEL_DESCRIPTION_H
#define LEVEL_DESCRIPTION_H

#include <wvu_swarm_std_msgs/vicon_point.h>
#include <wvu_swarm_std_msgs/map_levels.h>

namespace map_ns
{

// current levels that are included in the overall map
enum LEVEL
{
	NONE, TARGET, OBSTACLE, COMBINED
};

typedef enum LEVEL levelType;

/**
 * Calculates the z value at a given point for a given map
 *
 * @param ml is a map level that you are calculating a point on
 * @param loc is the current location of the robot
 *
 */
double calculate(wvu_swarm_std_msgs::map_level ml,
		wvu_swarm_std_msgs::vicon_point loc);
wvu_swarm_std_msgs::map_level combineLevels(wvu_swarm_std_msgs::map_level a,
		wvu_swarm_std_msgs::map_level b);

}
;
#endif
