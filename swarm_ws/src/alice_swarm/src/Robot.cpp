/*
 * Robot.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: smart6
 */

#include "alice_swarm/Robot.h"
#include "alice_swarm/Model.h"
#include "alice_swarm/VectorQueue.h"
#include "iostream"
Robot::Robot(){
name=10000;
}

Robot::Robot(AliceStructs::mail data)
{
	name = data.name;
	model = Model(data.name);
	vectorQueue = VectorQueue();
	model.addToModel(data);
	neighbors = data.neighbors;
}

void Robot::receiveMsg(AliceStructs::mail data)
{
	name = data.name;
	sid = data.sid;
	model = Model(data.name);
	vectorQueue = VectorQueue();
	model.addToModel(data);
	neighbors = data.neighbors;
}

AliceStructs::ideal Robot::generateIdeal()
{
	AliceStructs::ideal to_return = model.generateIdeal();
	//std::cout << name << std::endl;
	model.clear();
	vectorQueue.oneToQueue(to_return/*originally had "ideal", had no idea what was intended*/);
	return to_return;
}

AliceStructs::vel Robot::generateComp(std::vector<AliceStructs::ideal> ideals)
{

	if (model.rules.should_ignore== true)
	{
		return vectorQueue.createCompromise();
	}
	for (int i = 0; i < ideals.size(); i++)
	{
		for (int j = 0; j < neighbors.size(); j++)
		{
			if (ideals.at(i).name == neighbors.at(j).name && (sid==neighbors.at(j).sid))
			{

				ideals.at(i).dis = neighbors.at(j).dis;
				ideals.at(i).dir += neighbors.at(j).ang;
				vectorQueue.oneToQueue(ideals.at(i));
			}
		}
	}
	return vectorQueue.createCompromise();
}
