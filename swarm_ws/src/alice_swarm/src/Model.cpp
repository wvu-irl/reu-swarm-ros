#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include <iostream>

Model::Model(int _name)
{
	name = _name;
}

Model::sensorUpdate(AliceStructs::mail _toAdd)
{
	current_obstacles = _toAdd.obstacles;
	current_heat_map = _toAdd.heat_map;
}

Model::pass()
{
	//Dummy for the moment; will be updated at a later date
}

Model::forget()
{
	//See pass
}

string Model::getCollisionState()
{
	for (auto& obstacle : current_obstacles)
	{
		int dis = findLocalMin(obstacle);

	}
	return "Free";
}

int Model::findLocalMin(AliceStructs::obstacle)
{

}
