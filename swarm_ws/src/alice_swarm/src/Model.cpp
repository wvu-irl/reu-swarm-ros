#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include <iostream>
#include <ros/ros.h>

Model::Model()
{
}

Model::Model(int _name)
{
	name = _name;
}

void Model::addPolarVel(AliceStructs::vel &_vel1, AliceStructs::vel &_vel2)
{
	float x(_vel1.mag * cos(_vel1.dir) + _vel2.mag * cos(_vel2.dir));
	float y(_vel1.mag * sin(_vel1.dir) + _vel2.mag * sin(_vel2.dir));
	_vel1.mag = pow(pow(x, 2) + pow(y, 2), 0.5);
	_vel1.dir = atan2(y, x);
}
void Model::normalize(AliceStructs::vel &_vel)
{
	if (_vel.mag > 1)
		_vel.mag = 1;
}

void Model::dDriveAdjust(AliceStructs::vel &_vel) //assumes that vector has already been normalized
{
	if (_vel.dir > M_PI / 18 && _vel.dir < M_PI)
	{
		_vel.dir = M_PI / 18;
		_vel.mag = 0.01;
	} else if (_vel.dir > M_PI && _vel.dir < 35 * M_PI / 18)
	{
		_vel.dir = 35 * M_PI / 18;
		_vel.mag = 0.01;
	}

}

AliceStructs::ideal Model::generateIdeal()
{

	AliceStructs::ideal toReturn;
	AliceStructs::vel temp;
	temp.mag = 0;
	temp.dir = 0;
	rules.should_ignore = false;

	std::vector<AliceStructs::vel> ideal_list =
	{ //rules.dummy1(),
			rules.avoidObstacles(obstacles, 5, 360),
			rules.magnetAvoid(robots, 5),
			//rules.birdAvoid(robots, 5, 30),
			rules.maintainSpacing(robots, 2)
			};
	for (int i = 0; i < ideal_list.size(); i++)
	{
		if (ideal_list.at(i).mag > 0.01) {
			addPolarVel(temp, ideal_list.at(i));
		}
	}
	toReturn.pri = temp.mag;
	//normalize(temp);
	//dDriveAdjust(temp);
	toReturn.dir = temp.dir;
	toReturn.spd = temp.mag;
	//std::cout << temp.mag << " " << temp.dir << std::endl;
	toReturn.dis = 0;
	//(ideal_list.size()  + 1) / (float) (tolerance + 1);
	toReturn.name = name;
	return toReturn;

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
