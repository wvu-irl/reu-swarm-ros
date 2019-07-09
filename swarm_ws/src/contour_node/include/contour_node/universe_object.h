#ifndef UNIVERSE_H
#define UNIVERSE_H

#include <iostream>

#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include "level_object.h"

class Universe
{
private:
	wvu_swarm_std_msgs::map_levels overall_map;

public:
	Universe();

	void addEquation(wvu_swarm_std_msgs::obstacle equ);
	void addEquation(levelObject *equ);

	size_t numEquaitons();

	wvu_swarm_std_msgs::gaussian* findMsgByName(std::string);
	wvu_swarm_std_msgs::gaussian* findMsgByLocation(std::pair<double, double>);

	levelObject* findByName(std::string);
	levelObject* findByLocation(std::pair<double, double>);

	wvu_swarm_std_msgs::map_levels& getPublishable();

	explicit operator wvu_swarm_std_msgs::map_levels&();

	friend void operator+=(Universe&, levelObject*);
	friend void operator+=(Universe&, wvu_swarm_std_msgs::obstacle&);

	friend void operator-=(Universe&, std::string);
	friend void operator-=(Universe&, std::pair<double, double>);

	friend std::ostream& operator<<(Universe&, std::ostream&);
};

#endif
