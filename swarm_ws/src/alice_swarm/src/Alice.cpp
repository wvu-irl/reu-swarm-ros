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

#define DEBUG_generateVel 0

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
	for(auto& _rel_charger : _data.rel_chargerMail)
	{
			mail.rel_chargers.push_back(_rel_charger);
	}
	mail.abs_chargers = &(_chargers);
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
	mail.battery_lvl = _data.sensor_data.battery_level;
	mail.battery_state = (BatState) _data.sensor_data.battery_state;
	mail.energy = _data.energy;
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
<<<<<<< HEAD
	model.transformFir(model.goTo.x, model.goTo.y); //Shifts the frame of goTo to the current frame for calculations
	model = rules.stateLoop(model);
	//std::cout << model.goTo.x << " " << model.cur_pose.x << " - " << model.goTo.y << " " << model.cur_pose.y << std::endl;
	AliceStructs::vel to_return;
	to_return.mag = 1;
	//std::cout << "x: " << model.goTo.x << "y: " << model.goTo.y << std::endl;
	to_return.dir = atan2((model.goTo.y),(model.goTo.x));
	std::cout << to_return.dir << std::endl;

#if DEBUG_generateVel
	std::cout<<"alice executes: "<<rules.model.goTo.x<<","<<rules.model.goTo.y<<std::endl;//this is the rules model
	std::cout<<"alice executes: "<<model.goTo.x<<","<< model.goTo.y<<std::endl; //this is the alice model (not the same).
	std::cout<<"=================================================\n";
#endif

	model.transformCur(model.goTo.x, model.goTo.y); //Shifts the frame go goTo back to the first frame
=======
	model = rules.stateLoop(model);

	std::pair<float,float> cur_goTo = model.transformFir(model.goTo.x, model.goTo.y); //Shifts the frame of goTo to the current frame.
	model.goTo.x = cur_goTo.first;
	model.goTo.y = cur_goTo.second;

	AliceStructs::vel to_return;
	to_return.mag = 1;

	if(model.goTo.x == 0 && model.goTo.y == 0)
	{
		std::cout<<"===================mag zero================\n";
		to_return.mag = 0;
	}

	float theta = atan((model.goTo.y - model.cur_pose.y)/(model.goTo.x - model.cur_pose.x)) - model.cur_pose.heading;
	to_return.dir = fmod(atan2(model.goTo.y,model.goTo.x),2*M_PI);

#if DEBUG_generateVel
	std::cout<<"rules model: "<<rules.model.goTo.x<<","<<rules.model.goTo.y<<std::endl;//this is the rules model
	std::cout<<"alice model: "<<model.goTo.x<<","<< model.goTo.y<<std::endl; //this is the alice model (not the same).
	std::cout<<"to_return.dir: "<<to_return.dir<<std::endl;//this is the rules model
	std::cout <<"theta: "<<theta<< std::endl;
	std::cout <<"mag: "<<to_return.mag<< std::endl;
	std::cout<<"=================================================\n";
#endif

>>>>>>> c0a5e84... charge(), small fires, and energy.
	return to_return;
}
