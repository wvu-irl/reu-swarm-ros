/*
 * Robot.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: smart6
 */

#include "alice_swarm/Robot.h"

Robot::Robot(AliceStructs::mail data)
{
	model.addToModel(data);
	name = data.name;
	neighbors = data.neighbors;
}

AliceStructs::ideal Robot::generateIdeal()
{
	AliceStructs::ideal to_return = model.generateIdeal();
	model.clear();
	vector_queue.oneToQueue(to_return/*originally had "ideal", had no idea what was intended*/);
	return to_return;
}

AliceStructs::vel Robot::generateComp(std::vector<AliceStructs::ideal> ideals)
{
	for (int i = 0; i < ideals.size(); i++)
	{
		for (int j = 0; j < ideals.size(); j++)
		{
			if (ideals.at(i).name = neighbors.at(j).name)
			{
				ideals.at(i).dis = neighbors.at(j).dis;
				vector_queue.oneToQueue(ideals.at(i));
			}
		}
	}
	return vectorQueue.createCompromise();
}

}
