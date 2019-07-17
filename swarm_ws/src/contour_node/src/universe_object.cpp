#include <contour_node/universe_object.h>
#include <contour_node/gaussian_object.h>
#include <contour_node/level_description.h>

#include <sstream>

Universe::Universe()
{
	// much nothing
	// many wow
}

void Universe::addEquation(wvu_swarm_std_msgs::obstacle obs)
{
	while (overall_map.levels.size() <= obs.level) // making sure it is safe to add to a level
	{
		overall_map.levels.push_back(wvu_swarm_std_msgs::map_level()); // adding blank levels if they do not exist
	}
	bool found = false; // boolean for checking if the function already exists

	for (size_t i = 0;
			i < overall_map.levels.at(obs.level).functions.size() && !found; i++) // searching the overall map
	{
		if (overall_map.levels.at(obs.level).functions.at(i).name.compare(
				obs.characteristic.name) == 0) // checking for name equivalence
		{
			overall_map.levels.at(obs.level).functions[i] = obs.characteristic; // setting found function to new parameters
			found = true;
		}
	}

	if (!found) // not an old function
		overall_map.levels[obs.level].functions.push_back(obs.characteristic); // adding new

	if (obs.level != map_ns::COMBINED) // recursive check
	{
		// creating a copy for the combined level
		wvu_swarm_std_msgs::obstacle comb;
		obs.characteristic.amplitude *= 1 - (((obs.level + 1) % 2) * 2);
		comb.characteristic = obs.characteristic;
		comb.level = map_ns::COMBINED;

		// adding to the combined level
		addEquation(comb);
	}
}

void Universe::addEquation(levelObject *equ)
{
	this->addEquation(((gaussianObject*) equ)->getGaussianMessage()); // converting to a message then using the other function
}

wvu_swarm_std_msgs::gaussian* Universe::findMsgByName(std::string name)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus; // Preemptive pointer creation to match scope
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++) // searching each level
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++) // checking within the level
		{
			if (overall_map.levels[i].functions[j].name.compare(name) == 0) // comparing names
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j]; // setting pointer
			}
		}
	}

	if (found)
		return gaus;
	else
		return NULL;
}

wvu_swarm_std_msgs::gaussian* Universe::findMsgByLocation(
		std::pair<double, double> loc)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			std::pair<double, double> orig;
			orig.first = overall_map.levels[i].functions[j].ellipse.offset_x;
			orig.second = overall_map.levels[i].functions[j].ellipse.offset_y; // same as findMsgByName but with different conditional
			if (orig == loc)
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j];
			}
		}
	}
	if (found)
		return gaus;
	else
		return NULL;
}

levelObject* Universe::findByName(std::string name)
{
	bool found = false;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE; // this is the same as the message equivalent but level is tracked
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			if (overall_map.levels[i].functions[j].name.compare(name) == 0)
			{
				found = true;
				gaus = &overall_map.levels[i].functions[j];
				lev = (map_ns::LEVEL) i; // setting level for found function
			}
		}
	}

	if (!found)
		return NULL;

	// creating object
	wvu_swarm_std_msgs::obstacle obs;
	obs.characteristic = *gaus;
	obs.level = lev;
	gaussianObject *obj = new gaussianObject(obs);
	return obj;
}

levelObject* Universe::findByLocation(std::pair<double, double> loc)
{
	// look above; this is a copy paste
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

	if (!found)
		return NULL;

	wvu_swarm_std_msgs::obstacle obs;
	obs.characteristic = *gaus;
	obs.level = lev;
	gaussianObject *obj = new gaussianObject(obs);
	return obj;
}

levelObject* Universe::findWithinRadius(std::pair<double, double> loc, double radius)
{
	bool found = false;
        radius *= radius; // Square now to not need sqrt later
        double shortestDistance = radius;
	wvu_swarm_std_msgs::gaussian *gaus;
	map_ns::LEVEL lev = map_ns::NONE;
        
	for (size_t i = 0; i < overall_map.levels.size() && !found; i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			std::pair<double, double> orig;
			orig.first = overall_map.levels[i].functions[j].ellipse.offset_x;
			orig.second = overall_map.levels[i].functions[j].ellipse.offset_y;
                        
                        double distance = pow(orig.first - loc.first, 2) + pow(orig.second - loc.second, 2);
                        
			if (distance < shortestDistance)
			{
                                shortestDistance = distance;
				found = true;
				gaus = &overall_map.levels[i].functions[j];
				lev = (map_ns::LEVEL) i;
			}
		}
	}

        if(found)
        {
            wvu_swarm_std_msgs::obstacle obs;
            obs.characteristic = *gaus;
            obs.level = lev;
            gaussianObject *obj = new gaussianObject(obs);
            return obj;
        }
        else
        {
            return nullptr;
        }
}

wvu_swarm_std_msgs::map_levels& Universe::getPublishable()
{
	return overall_map; // this is an accessor
}

size_t Universe::numEquaitons()
{
	size_t count = 0;
	for (size_t i = 0; i < overall_map.levels.size(); i++) // going through all the levels
	{
		count += overall_map.levels[i].functions.size(); // summing all the sizes of each level
	}
	return count;
}

void Universe::foreach(
		std::function<void(wvu_swarm_std_msgs::gaussian*, int)> funk)
{
	for (size_t i = 0; i < overall_map.levels.size(); i++)
	{
		for (size_t j = 0; j < overall_map.levels[i].functions.size(); j++)
		{
			funk(&overall_map.levels[i].functions[j], (int) i);
		}
	}
}

Universe::operator wvu_swarm_std_msgs::map_levels&()
{
	return overall_map; // cast acts the same way as the get publishable
}

void operator+=(Universe &uni, levelObject *obj)
{
	uni.addEquation(obj); // operator acts as a macro for this function
}

void operator+=(Universe &uni, wvu_swarm_std_msgs::obstacle &obj)
{
	uni.addEquation(obj); // same as ^^
}

void operator-=(Universe &uni, std::string name)
{
	// finds object like before
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
		// erasing from map
		overall_map.levels[i].functions.erase(
				overall_map.levels[i].functions.begin() + j);
	}

}

void operator-=(Universe &uni, std::pair<double, double> loc)
{
	// copy paste of ^^
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
	}
}

std::ostream& operator<<(std::ostream &stream, Universe &uni)
{
	// cout operator to print universe to the console
	std::stringstream ss;
	ss << "Universe:\n{\n" << uni.getPublishable() << "\n}";
	return stream << ss.str();
}
