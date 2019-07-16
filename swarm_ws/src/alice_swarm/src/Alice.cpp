#include "alice_swarm/Alice.h"
#include <alice_swarm/Rules.h>
#include "alice_swarm/Model.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/alice_mail.h"
#include "wvu_swarm_std_msgs/gaussian.h"
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include "wvu_swarm_std_msgs/map.h"

#include "alice_swarm/aliceStructs.h"

#include <iostream>

Alice::Alice()
{
	name = 0;
}

AliceStructs::mail Alice::packageData(wvu_swarm_std_msgs::alice_mail &_data, 
		std::vector<wvu_swarm_std_msgs::charger> &_chargers, std::vector<float> &_priority)
{
	AliceStructs::mail mail;
	//------Updates all vectors in mail -----------
	for (auto& _obstacle : _data.obsMail)
	{
		AliceStructs::obj obstacle;
		obstacle.x_off = _obstacle.offset_x;
		obstacle.y_off = _obstacle.offset_y;
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
		tar.z = 1;//targets don't have this value?
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
//	for(int i = 0; i < _priority.size(); i++)
//	{
//			mail.priority.push_back(_priority.at(i));
//	}
	mail.chargers = &(_chargers);
	mail.priority = &(_priority);
//--------------------------------------------

	//----Updates simple attributes----------
	mail.xpos=_data.x;
	mail.ypos=_data.y;
	mail.contVal=_data.contVal;
	mail.heading=_data.heading;
	mail.name = _data.name;
	mail.vision=_data.vision;
	mail.time=_data.time;
	mail.battery_lvl = _data.battery_lvl;
	//-----------------------------------------

	return mail;
}

void Alice::updateModel(wvu_swarm_std_msgs::alice_mail &_data, std::vector<wvu_swarm_std_msgs::map> &_maps,  std::vector<int> &_ids,
		std::vector<wvu_swarm_std_msgs::charger> &_chargers, std::vector<float> &_priority)
{
	name=_data.name;
	AliceStructs::mail mail = packageData(_data, _chargers, _priority);
	model.archiveAdd(mail);
	model.sensorUpdate(mail);
	model.receiveMap(_maps,_ids);
	model.forget();
}

AliceStructs::vel Alice::generateVel() //implements the rules set
{
	std::cout << "loop started" << std::endl;
	rules.stateLoop(model);
	//std::cout << model.goTo.x << " " << model.cur_pose.x << " - " << model.goTo.y << " " << model.cur_pose.y << std::endl;
	AliceStructs::vel to_return;
	to_return.mag = 1;
	//std::cout << "x: " << model.goTo.x << "y: " << model.goTo.y << std::endl;
	to_return.dir = atan((model.goTo.y - model.cur_pose.y)/(model.goTo.x - model.cur_pose.x)) - model.cur_pose.heading;
	std::cout << to_return.dir << std::endl;
	return to_return;
}
