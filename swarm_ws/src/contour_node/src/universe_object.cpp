#include <contour_node/universe_object.h>
#include <contour_node/gaussian_object.h>
#include <contour_node/level_description.h>

#include <sstream>

Universe::Universe() :
		numEqs_(0)
{
}

void Universe::addEquation(wvu_swarm_std_msgs::obstacle obs)
{
	while (overall_map.levels.size() <= obs.level)
	{
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
		obs.characteristic.amplitude *= 1 - (((obs.level + 1) % 2) * 2);
		comb.characteristic = obs.characteristic;
		comb.level = map_ns::COMBINED;

		addEquation(comb);
	}
	numEqs_++;
}

void Universe::addEquation(levelObject *equ)
{
	this->addEquation(((gaussianObject*) equ)->getGaussianMessage());
}

wvu_swarm_std_msgs::gaussian* Universe::findMsgByName(std::string name)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			if (overall_map.levels[i].functions[j].name.compare(name) == 0)
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j];
				lev = (map_ns::LEVEL) i;
			}
		}
	}

	return gaus;
}

wvu_swarm_std_msgs::gaussian* Universe::findMsgByLocation(
		std::pair<double, double> loc)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			std::pair<double, double> orig;
			orig.first = overall_map.levels[i].functions[j].ellipse.offset_x;
			orig.second = overall_map.levels[i].functions[j].ellipse.offset_y;
			if (orig == loc)
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j];
				lev = (map_ns::LEVEL) i;
			}
		}
	}
	return gaus;
}

levelObject* Universe::findByName(std::string name)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			if (overall_map.levels[i].functions[j].name.compare(name) == 0)
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j];
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

levelObject* Universe::findByLocation(std::pair<double, double> loc)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			std::pair<double, double> orig;
			orig.first = overall_map.levels[i].functions[j].ellipse.offset_x;
			orig.second = overall_map.levels[i].functions[j].ellipse.offset_y;
			if (orig == loc)
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j];
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

wvu_swarm_std_msgs::map_levels& Universe::getPublishable()
{
	return overall_map;
}

size_t Universe::numEquaitons()
{
	return numEqs_;
}

Universe::operator wvu_swarm_std_msgs::map_levels&()
{
	return overall_map;
}

void operator+=(Universe &uni, levelObject *obj)
{
	uni.addEquation(obj);
}

void operator+=(Universe &uni, wvu_swarm_std_msgs::obstacle &obj)
{
	uni.addEquation(obj);
}

void operator-=(Universe &uni, std::string name)
{
	wvu_swarm_std_msgs::map_levels &overall_map = uni.getPublishable();
	bool found = false;
	size_t i = 0;
	size_t j = 0;
	for (; i < overall_map.levels.size() && !found; i++)
	{
		for (; j < overall_map.levels[i].functions.size(); j++)
		{
			if (overall_map.levels[i].functions[j].name.compare(name) == 0)
			{
				found = true;
			}
		}
	}
	if (found)
	{
		overall_map.levels[i].functions.erase(
				overall_map.levels[i].functions.begin() + j);
		uni.numEqs_--;
	}

}

void operator-=(Universe &uni, std::pair<double, double> loc)
{
	wvu_swarm_std_msgs::map_levels &overall_map = uni.getPublishable();
	bool found = false;
	size_t i = 0;
	size_t j = 0;
	for (; i < overall_map.levels.size() && !found; i++)
	{
		for (; j < overall_map.levels[i].functions.size(); j++)
		{
			std::pair<double, double> orig;
			orig.first = overall_map.levels[i].functions[j].ellipse.offset_x;
			orig.second = overall_map.levels[i].functions[j].ellipse.offset_y;
			if (orig == loc)
			{
				found = true;
			}
		}
	}
	if (found)
	{
		overall_map.levels[i].functions.erase(
				overall_map.levels[i].functions.begin() + j);
		uni.numEqs_--;
	}
}

std::ostream& operator<<(Universe &uni, std::ostream &stream)
{
	std::stringstream ss;
	ss << "Universe:\n{\n" << uni.getPublishable() << "\n}";
	return stream << ss.str();
}
