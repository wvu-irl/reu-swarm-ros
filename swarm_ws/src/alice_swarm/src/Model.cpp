#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include "wvu_swarm_std_msgs"
#include <iostream>

Model::Model(int _name)
{
	name = _name;
}

model::sensorUpdate(aliceStructs::mail _toAdd)
{
	for (auto& obstacle : _toAdd.obstacles) {
		obstacles.push_back(obstacle);
	}

}

Model::pass()
{
	//Dummy for the moment; will be updated at a later date
}

Model::forget()
{
	//See pass
}
