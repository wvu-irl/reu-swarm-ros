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
	model.addToModel(data);
	neighbors = data.neighbors;
}

AliceStructs::ideal Robot::generateIdeal()
{
	AliceStructs::ideal to_return = model.generateIdeal();
	model.clear();
	vectorQueue.oneToQueue(to_return/*originally had "ideal", had no idea what was intended*/);
	return to_return;
}

AliceStructs::vel Robot::generateComp(std::vector<AliceStructs::ideal> ideals)
{

	for (int i = 0; i < ideals.size(); i++)
	{
		for (int j = 0; j < neighbors.size(); j++)
		{
			if (ideals.at(i).name = neighbors.at(j).name)
			{
				ideals.at(i).dis = neighbors.at(j).dis;
				vectorQueue.oneToQueue(ideals.at(i));
			}
		}
	}
	return vectorQueue.createCompromise();
}
