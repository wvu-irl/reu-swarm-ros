#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include <ros/ros.h>

class Model
{

Model::Model(int _name){
	name = _name;
}


	AliceStructs::ideal Model::generateIdeal()
{
	ideal toReturn;
	toReturn.dir = -1;
	int tolerance = 1;
	while ((toReturn.dir == -1) && ros::ok)
	{
		std::list <vel> ideal_list = {rules.avoidObstacles(obstacles, tolerance),
			rules.avoidRobots(robots, tolerance),
			rules.maintainSpacing(robots, tolerance)};
		for (int i = 0; i < ideal_list.size(); i++)
		{
			if (ideal_list.get(i).dis != -1)
			{
				to_return.dir = ideal_list.get(i).dir;
				to_return.spd = ideal_list.get(i).mag;
				to_return.dis = 0;
				to_return.pri = (ideal_list.size() - i + 1)/(tolerance + 1);
				to_return.name = name;
				return to_return;
			}
		}
		tolerance += 1;
	}
}

void Model::addToModel(mail toAdd)
{
	for (auto& item : toAdd.obstacles)
	{
		obstacles.insert(item)
	}
	for (auto& item : toAdd.neighbors)
	{
		neighbors.insert(item)
	}
	for (auto& item : toAdd.targets)
	{
		targets.insert(item)
	}
}

void Model::clear()
{
	obstacles.clear()
	robots.clear()
	targets.clear()
}
		
