/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

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
	model = Model();
}

AliceStructs::mail Alice::packageData(wvu_swarm_std_msgs::alice_mail &_data, 
		std::vector<wvu_swarm_std_msgs::charger> *_chargers, std::vector<float> &_priority)
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
	mail.abs_chargers = _chargers;
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
	mail.energy = _data.energy;
	//-----------------------------------------

	return mail;
}

void Alice::updateModel(wvu_swarm_std_msgs::alice_mail &_data, std::vector<wvu_swarm_std_msgs::map> &_maps,  std::vector<int> &_ids,
		std::vector<wvu_swarm_std_msgs::charger> *_chargers, std::vector<float> &_priority)
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
	rules.stateLoop(model);

	std::pair<float,float> cur_go_to = model.transformFir(model.goTo.x, model.goTo.y); //Shifts the frame of goTo to the current frame.


	AliceStructs::vel to_return;
	to_return.mag = 1;

//	if(rules.calcDis(cur_go_to.first, cur_go_to.second, 0, 0)<0.9) //gives a tolerance range
	if(cur_go_to.first == 0 && cur_go_to.second == 0) //gives a tolerance range
	{
#if DEBUG_generateVel
		std::cout<<"===================mag zero================\n";
#endif
		to_return.mag = 0;
	}

	to_return.dir = fmod(atan2(cur_go_to.second,cur_go_to.first),2*M_PI);
#if DEBUG_generateVel
//	std::cout<<"rules model: "<<rules.cur_go_to.first<<","<<rules.cur_go_to.second<<std::endl;//this is the rules model
//	std::cout<<"alice model: "<<cur_go_to.first<<","<< cur_go_to.second<<std::endl; //this is the alice model (not the same).
	std::cout<<"to_return.dir: "<<to_return.dir<<std::endl;//this is the rules model
	std::cout <<"mag: "<<to_return.mag<< std::endl;
//	std::cout<<"=================================================\n";
#endif

	return to_return;
}
