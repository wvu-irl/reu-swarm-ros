#include "alice_swarm/Alice.h"
#include "alice_swarm/Rules.h"
#include "alice_swarm/Model.h"
#include "wvu_swarm_std_msgs"
#include <iostream>

Alice::Alice(AliceStructs::mail _data)
{
	name = _data.name;
	model = Model(_data.name);
	rules = Rules(model);
	model.updateModel(_data);
}

AliceStructs::mail Alice::packageData(wvu_swarm_std_msgs::mail _data)
{
	AliceStructs::mail mail;
	for (auto& _obstacle : _data.obsMail)
	{
		AliceStructs::obj obstacle;
		obstacle.x_rad = _obstacle.x_rad;
		obstacle.y_rad = _obstacle.y_rad;
		obstacle.theta_offset = _obstacle.theta_offset;
		mail.obstacles.push_back(obstacle);
	}
	for (auto& _neighbor : _data.neighborMail)
	{
		AliceStructs::neighbor neighbor;
		neighbor.dir = _neighbor.dir;
		neighbor.dis = _neighbor.dis;
		neighbor.ang = _neighbor.ang;
		neighbor.name = _neighbor.name;
		mail.neighbors.push_back(neighbor);
	}
	for (auto& _tar : _data.targetMail)
	{
		AliceStructs::tar tar;
		tar.x = _tar.x;
		tar.y = _tar.y;
		tar.z = _tar.z;
		mail.neighbors.push_back(neighbor);
	}
	for (auto& _flow : flowMail)
	{
		AliceStructs::flow flow;
		flow.dir = _flow.dir;
		flow.dis = _flow.dis;
		flow.pri = _flow.pri;
		flow.spd = _flow.spd;
		mail.flows.push_back(flow);
	}
	mail.name = name;
	return mail
}

void Alice::updateModel(wvu_swarm_std_msgs::mail _data)
{
	mail = packageData(_data);
	model.updateModel(mail);
}

AliceStructs::vel generateVel()
{
	rules.generateVel();
}
