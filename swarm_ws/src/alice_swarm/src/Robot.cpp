/*
 * Robot.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: smart6
 */

#include "Robot.h"

class Robot
{

	Robot(mail data)
	{
		model.addToModel(data);
		name = data.name;
		neighbors = data.neighbors;
	}

	ideal Robot::generateIdeal()
	{
		ideal to_return = model.generateIdeal();
		model.clear();
		vector_queue.oneToQueue(ideal);
		return to_return;
	}

	vel Robot::generateComp(std::vector <ideal> ideals)
	{
		for (int i = 0; i < ideals.size(); i++)
		{
			for (int j = 0; j < ideals.size(); j++)
			{
				if (ideals.at(i).name = neighbors.at(j).name) {
					ideals.at(i).dis = neighbors.at(j).dis;
					vector_queue.oneToQueue(ideals.get(i));
				}
			}
		}
		return vectorQueue.createCompromise();
	}

}
