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
Robot::Robot(){ //dummy constructor
name=10000;
}

Robot::Robot(AliceStructs::mail data)
{
	//copies fields of the mail to fields of robot
	name = data.name;
	model = Model(data.name,data.sid);
	vectorQueue = VectorQueue();
	model.addToModel(data);
	neighbors = data.neighbors;
}

void Robot::receiveMsg(AliceStructs::mail data)
{
	//copies fields of mail to the field of robot
	name = data.name;
	sid = data.sid;
	model = Model(data.name,data.sid);
	vectorQueue = VectorQueue();
	model.addToModel(data);
	neighbors = data.neighbors;
}

AliceStructs::ideal Robot::generateIdeal()
{
	//calls generateIdeal from model
	AliceStructs::ideal to_return = model.generateIdeal();
	model.clear();
	//adds the ideal to the vector queue
	vectorQueue.oneToQueue(to_return);
	return to_return;
}

AliceStructs::vel Robot::generateComp(std::vector<AliceStructs::ideal> ideals)
{

	if (model.rules.should_ignore== true) //if we don't want to care about anyone else's ideals
	{
		return vectorQueue.createCompromise();
	}
	for (int i = 0; i < ideals.size(); i++)
	{
		for (int j = 0; j < neighbors.size(); j++)
		{
			if (ideals.at(i).name == neighbors.at(j).name && sid==neighbors.at(j).sid ) // if the ideal is from a neighbor with the same swarm id
			{

				ideals.at(i).dis = neighbors.at(j).dis;
				ideals.at(i).dir += neighbors.at(j).ang; //Makes the angle relative to the bot
				vectorQueue.oneToQueue(ideals.at(i));
			}
		}
	}
	return vectorQueue.createCompromise();
}
