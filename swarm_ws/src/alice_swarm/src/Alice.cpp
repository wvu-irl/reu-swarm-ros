#include "alice_swarm/Alice.h"
#include <alice_swarm/Rules.h>
#include "alice_swarm/Model.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/alice_mail.h"
#include "wvu_swarm_std_msgs/gaussian.h"
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include "alice_swarm/aliceStructs.h"

#include <iostream>

Alice::Alice()
{
	name = 0;
}

Alice::Alice(wvu_swarm_std_msgs::alice_mail _data)
{
	name = _data.name;
	model = Model(_data.name);
	updateModel(_data);
	rules = Rules(model);
//	(_data);
}

AliceStructs::mail Alice::packageData(wvu_swarm_std_msgs::alice_mail _data)
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
		neighbor.x = _neighbor.x;
		neighbor.y = _neighbor.y;
		neighbor.ang = _neighbor.ang;
		neighbor.name = _neighbor.name;
		mail.neighbors.push_back(neighbor);
	}
	for (auto& _tar : _data.targetMail)
	{
		AliceStructs::pnt tar;
		tar.x = _tar.x;
		tar.y = _tar.y;
		//tar.z = _tar.z;
		mail.targets.push_back(tar);
	}
	for (auto& _flow : _data.flowMail)
	{
		AliceStructs::flow flow;
		flow.x = _flow.x;
		flow.y = _flow.y;
		flow.pri = _flow.pri;
		flow.spd = _flow.spd;
		mail.flows.push_back(flow);
	}
	mail.name = _data.name;
	return mail;
}

void Alice::updateModel(wvu_swarm_std_msgs::alice_mail _data)
{
	AliceStructs::mail mail = packageData(_data);
	model.sensorUpdate(mail);
}

AliceStructs::vel Alice::generateVel()
{
	AliceStructs::vel to_return = rules.stateLoop();
	std::cout << "mag: " << to_return.mag << " - dir: " << to_return.dir << std::endl;
	return to_return;
}
