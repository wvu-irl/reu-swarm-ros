#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include <iostream>[
#include <ros/ros.h>

Model::Model(){
}

Model::Model(int _name){
	name = _name;
}


AliceStructs::ideal Model::generateIdeal()
{
	AliceStructs::ideal toReturn;
	toReturn.spd = -1;
	int tolerance = 1;
	while ((toReturn.spd == -1) && ros::ok)
	{
		std::vector <AliceStructs::vel> ideal_list = {rules.avoidObstacles(obstacles, tolerance),
			rules.avoidRobots(robots, tolerance),
			rules.maintainSpacing(robots, tolerance)};
		for (int i = 0; i < ideal_list.size(); i++)
		{
			if (ideal_list.at(i).mag != -1)
			{
				toReturn.dir = ideal_list.at(i).dir;
				toReturn.spd = ideal_list.at(i).mag;
				toReturn.dis = 0;
				toReturn.pri = (ideal_list.size() - i + 1)/(float) (tolerance + 1);

				toReturn.name = name;
				return toReturn;
			}
		}
		tolerance += 1;
	}
}

void Model::addToModel(AliceStructs::mail toAdd)
{
	for (auto& item : toAdd.obstacles)
	{
		obstacles.push_back(item);
	}
	for (auto& item : toAdd.neighbors)
	{
		robots.push_back(item);
	}
	for (auto& item : toAdd.targets)
	{
		targets.push_back(item);
	}
}

void Model::clear()
{
	obstacles.clear();
	robots.clear();
	targets.clear();
}

