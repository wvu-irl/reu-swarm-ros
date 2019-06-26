#include "alice_swarm/Robot.h"
#include "alice_swarm/Model.h"
#include "alice_swarm/VectorQueue.h"
#include "iostream"

/*
 * Dummy constructor, exists only for compiler reasons
 */
Robot::Robot()
{
	name=10000;
}

/*
 *	Initializes the model and passes data from the sensors to the model
 */
Robot::Robot(AliceStructs::mail data)
{
	name = data.name;
	model = Model(data.name,data.sid);
	vectorQueue = VectorQueue();
	model.addToModel(data);
	neighbors = data.neighbors;
}

/*
 * Passes data from the sensors to the sensors to the model
 */
void Robot::receiveMsg(AliceStructs::mail data)
{
	name = data.name;
	sid = data.sid;
	model = Model(data.name,data.sid);
	vectorQueue = VectorQueue();
	model.addToModel(data);
	neighbors = data.neighbors;
}

/*
 * Generates an ideal vector using the model, then passes that vector to the VectorQueue
 */
AliceStructs::ideal Robot::generateIdeal()
{
	AliceStructs::ideal to_return = model.generateIdeal();
	model.clear();
	vectorQueue.oneToQueue(to_return);
	return to_return;
}

/*
 * This method first passes ideal vectors from neighbors in its swarm to its VectorQueue, then
 * uses the updated queue to generate a velocity vector to be passed to the low-level controls
 */
AliceStructs::vel Robot::generateComp(std::vector<AliceStructs::ideal> ideals)
{

	if (model.rules.should_ignore== true) //If the robot needs to ignore the desires of the other robots
	{
		return vectorQueue.createCompromise();
	}
	for (int i = 0; i < ideals.size(); i++)
	{
		for (int j = 0; j < neighbors.size(); j++)
		{
			if (ideals.at(i).name == neighbors.at(j).name && sid==neighbors.at(j).sid )
			{

				ideals.at(i).dis = neighbors.at(j).dis;
				ideals.at(i).dir += neighbors.at(j).ang; //Makes the neighbor's angles relative to this bots angle
				vectorQueue.oneToQueue(ideals.at(i));
			}
		}
	}
	return vectorQueue.createCompromise();
}
