#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/obs_point_mail.h"
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include <iostream>

Model::Model(int _name)
{
	name = _name;
}

model::sensorUpdate(aliceStructs::mail _toAdd)
{
	for (auto& obstacle : _toAdd.obstacles)
	{
		obstacles.push_back(obstacle);
	}
	for (auto& flow : _toAdd.flows)
	{
		flows.push_back(flow);
	}
	for (auto& neighbor : _toAdd.neighbors)
	{
		neighbors.push_back(neighbor);
	}
	for (auto& tar : _toAdd.targets)
	{
		targets.push_back(tar);
	}
	current_level.z = _toAdd.level;
}

Model::pass()
{
	//Dummy for the moment; will be updated at a later date
}

Model::forget()
{
	//See pass
}
