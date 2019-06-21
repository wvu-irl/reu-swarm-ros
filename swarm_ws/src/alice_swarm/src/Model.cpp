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

void Model::addIdeal(AliceStructs::ideal &_ideal1, AliceStructs::ideal &_ideal2) // ideals are always polar
{
	_ideal1.dir = (_ideal1.dir * _ideal1.pri + _ideal2.pri * _ideal2.dir)/(_ideal1.pri + _ideal2.pri);
	_ideal1.pri += _ideal2.pri;
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

	AliceStructs::ideal to_return;
	//AliceStructs::ideal temp;
	to_return.dir = 0;
	to_return.pri = 0;
	rules.should_ignore = false;

	std::vector<AliceStructs::ideal> ideal_list =
	{ //rules.dummy1(),
			rules.followFlow(flows, 16),
			rules.goToTarget(targets, 16),
			//rules.avoidObstacles(obstacles, 16),
			//rules.magnetAvoid(robots, 16),
			//rules.birdAvoid(robots, 16),
			//rules.maintainSpacing(robots, 16)
			};
	for (int i = 0; i < ideal_list.size(); i++)
	{
		addIdeal(to_return, ideal_list.at(i));
	}
	//if (to_return.dir > 2*M_PI) std::cout << to_return.dir << std::endl;
	to_return.name = name;
	std::cout << name << std::endl;
	return to_return;

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
	for (auto& item : toAdd.flows)
	{
		flows.push_back(item);
	}
}

void Model::clear()
{
	obstacles.clear();
	robots.clear();
	targets.clear();
}
