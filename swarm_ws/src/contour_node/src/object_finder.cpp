#include <contour_node/object_finder.h>
#include <contour_node/gaussian_object.h>

bool operator==(std::pair<double, double> a, std::pair<double, double> b)
{
	return a.first == b.first && a.second == b.second;
}

levelObject* map_ns::findByName(wvu_swarm_std_msgs::map_levels map, std::string name)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
	for (size_t i = 0; i < map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < map.levels[i].functions.size(); j++)
		{
			if (map.levels[i].functions[j].name.compare(name) == 0)
			{
				found = true;
				gaus = &map.levels[i].functions[j];
				lev = (map_ns::LEVEL) i;
			}
		}
	}

	wvu_swarm_std_msgs::obstacle obs;
	obs.characteristic = *gaus;
	obs.level = lev;
	gaussianObject *obj = new gaussianObject(obs);
	return obj;
}

levelObject* map_ns::findByLocation(wvu_swarm_std_msgs::map_levels map,
		std::pair<double, double> loc)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
	for (size_t i = 0; i < map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < map.levels[i].functions.size(); j++)
		{
			std::pair<double, double> orig;
			orig.first = map.levels[i].functions[j].ellipse.offset_x;
			orig.second = map.levels[i].functions[j].ellipse.offset_y;
			if (orig == loc)
			{
				found = true;
				gaus = &map.levels[i].functions[j];
				lev = (map_ns::LEVEL) i;
			}
		}
	}

	wvu_swarm_std_msgs::obstacle obs;
	obs.characteristic = *gaus;
	obs.level = lev;
	gaussianObject *obj = new gaussianObject(obs);
	return obj;
}

size_t map_ns::numEquaitons(wvu_swarm_std_msgs::map_levels map)
{
	size_t count = 0;
	for (size_t i = 0;i < map.levels.size();i++)
	{
		count += map.levels[i].functions.size();
	}
	return count;
}
