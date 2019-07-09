#ifndef OBJ_FIND_H
#define OBJ_FIND_H

#include <stdlib.h>

#include "level_description.h"
#include <wvu_swarm_std_msgs/map_levels.h>
#include <string>

#include "level_object.h"

namespace map_ns
{
size_t numEquaitons(wvu_swarm_std_msgs::map_levels);

levelObject* findByName(wvu_swarm_std_msgs::map_levels, std::string);
levelObject* findByLocation(wvu_swarm_std_msgs::map_levels,
		std::pair<double, double>);
}
#endif
