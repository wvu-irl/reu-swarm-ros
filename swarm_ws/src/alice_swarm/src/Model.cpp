#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/gaussian.h"
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include <iostream>

Model::Model()
{
	name = 0;
}

Model::Model(int _name)
{
	name = _name;
}

void Model::clear()
{
	obstacles.clear();
	flows.clear();
	neighbors.clear();
	targets.clear();
}
void Model::archiveAdd(AliceStructs::mail &_toAdd)
{
	float vision = _toAdd.vision - 0; // make number larger if we want to account for moving things, will be buggy tho

	//placing objects that exited fov
	for (auto& obstacle : obstacles)
	{
		if (pow(pow(obstacle.x_off - _toAdd.xpos, 2) + pow(obstacle.y_off - _toAdd.ypos, 2), 0.5) > vision)
		{
			obstacle.time = time;
			obstacle.observer_name = name;
			archived_obstacles.push_back(obstacle);
		}

	}
	for (auto& tar : targets)
	{
		if (pow(pow(tar.x - _toAdd.xpos, 2) + pow(tar.y - _toAdd.ypos, 2), 0.5) > vision)
		{
			tar.time = time;
			tar.observer_name = name;
			archived_targets.push_back(tar);
		}
	}

	if (time.nsec % 10 == 0) //so we don't record literally every time step
	{
		AliceStructs::pose temp(cur_pose);
		temp.observer_name=name;
		archived_contour.push_back(temp);
	}
}

void Model::sensorUpdate(AliceStructs::mail &_toAdd)
{
	clear();
	cur_pose.x = _toAdd.xpos;
	cur_pose.y = _toAdd.ypos;
	cur_pose.z = _toAdd.level;
	cur_pose.heading = _toAdd.heading;
	name = _toAdd.name;
	time = _toAdd.time;
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
}

void Model::pass()
{
	//Dummy for the moment; will be updated at a later date
}

void Model::forget()
{
	//See pass
}
