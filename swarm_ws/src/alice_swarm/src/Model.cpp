#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/gaussian.h"
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include <iostream>

Model::Model(int _name)
{
	name = _name;
}

void Model::sensorUpdate(AliceStructs::mail _toAdd)
{
	for (auto& obstacle : _toAdd.obstacles)
	{
		current_obstacles.push_back(obstacle);
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
	cur_pose.z = _toAdd.level;
}

void Model::pass()
{
	//Dummy for the moment; will be updated at a later date
}

void Model::forget()
{
	//See pass
}
