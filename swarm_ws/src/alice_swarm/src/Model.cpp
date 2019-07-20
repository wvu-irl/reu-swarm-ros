#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include <iostream>
#include <ros/ros.h>

/*
 * Dummy constructor for the compiler
 */
Model::Model()
{
}

Model::Model(int _name, int _sid)
{
	name = _name;
	sid = _sid;
	rules = Rules(_sid);
}

/*
 * Generates an ideal vector from a given set of rules
 * To adjust the rules, simply comment and uncomment them
 * The second parameter each rules gets passed is it's strength, or to what extent the robot will priorotize it
 * For more info, see rules.cpp
 */
AliceStructs::ideal Model::generateIdeal()
{
	AliceStructs::ideal to_return;
	to_return.dir = 0;
	to_return.pri = 0.001;
	to_return.spd = 0.001;
	rules.should_ignore = true;

	std::vector<AliceStructs::ideal> ideal_list =
	{ //
//			rules.followFlow(flows, 16),
//			rules.goToTarget(targets, 0.5),
			rules.avoidObstacles(obstacles, 16),
			rules.magnetAvoid(robots, 16),
			//rules.birdAvoid(robots, 16),
	//	rules.maintainSpacing(robots, 16, 50)
	    };
	for (int i = 0; i < ideal_list.size(); i++)
	{
		//Prints out what each rule wants tells the robot to do. Is on one line to make commenting it out easier
	//	std::cout << "dir: " << ideal_list.at(i).dir << " spd: " << ideal_list.at(i).spd << " pri: " << ideal_list.at(i).pri << std::endl;
		to_return = rules.addIdeals(to_return, ideal_list.at(i));
	}
	to_return.name = name;
	return to_return;

}

/*
 * Adds information to the model
 */
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

/*
 * Clears the model to prevent memory issues
 */
void Model::clear()
{
	obstacles.clear();
	robots.clear();
	targets.clear();
	flows.clear();
}
