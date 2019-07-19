#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include <map>
#include <string>
#include <math.h>
#include <iostream>
#include <bits/stdc++.h>

#define DEBUG_schange 1

Rules::Rules()
{
	state = REST;
}

Rules::Rules(Model _model) :
		model(_model)
{
	state = REST;
}

//===================================================================================================================

Model Rules::stateLoop(Model &_model)
{
	model = _model; //do not comment this out. Doing so causes Ragnarok.
	std::pair<float,float> cur_goTo = model.transformFir(model.goTo.x, model.goTo.y); //Shifts the frame of goTo to the current frame for calculations
	model.goTo.x = cur_goTo.first;
	model.goTo.y = cur_goTo.second;

	if (true) //shouldLoop())
	{
		std::vector<AliceStructs::pnt> go_to_list;
		go_to_list.push_back(goToTar());
//		go_to_list.push_back(charge());
		// add other rules here
		float temp = -1;
		for (auto& rule : go_to_list)
		{
			//std::cout<<"the rule z is: "<<rule.z<<std::endl;
			if (rule.z > temp)
			{
				temp = rule.z;
				model.goTo.x = rule.x;
				model.goTo.y = rule.y;
				//std::cout<<"(x,y): "<<model.goTo.x<<","<<model.goTo.y<<std::endl;
			}
		}
		avoidCollisions();
		std::cout << "theta: "<<atan2(model.goTo.y, model.goTo.x) << std::endl;
		//std::cout<<"(x,y): "<<model.goTo.x<<","<<model.goTo.y<<std::endl;
		std::cout<<"-----------------------------------"<<std::endl;
	}
	std::pair<float,float> first_goTo = model.transformCur(model.goTo.x, model.goTo.y); //Shifts the frame of goTo back to the first frame
	model.goTo.x = first_goTo.first;
	model.goTo.y = first_goTo.second;

	return model;
}

//===================================================================================================================

bool Rules::shouldLoop()
{
	bool result = false;
	if (calcDis(model.goTo.x, model.goTo.y, 0, 0) < model.SIZE/2)
	{
		result = true;
	} /* to implement
	else if (checkCritical())
	{
		result = true;
	}
	else if (notConforming())
	{
		result = true;
	} */
	return true;
//	return result;
}

//===================================================================================================================

void Rules::avoidCollisions()
{
	bool checker = true;
	//while (checker)
	//{
	model.goTo = findPath(model.goTo, findDeadZones());
		//checker = avoidNeighbors();
	//}
}

void Rules::explore()
{
	AliceStructs::pose best;
	std::pair<float, float> sum(0,0);
	for (auto& contour : model.archived_contour)
	{
		std::pair<float, float> temp = model.transformFir(contour.x,contour.y); //transforms to the current frame
		float dist =  calcDis(0, 0, contour.x, contour.y);
		if (dist <model.vision) //simplistic, just adds the vectors together and goes the opposite way
		{
			sum.first+= temp.first;
			sum.second+=temp.second;//some form of priority assigned by confidence TBD
		}
	}
	final_vel.dir = atan2(-sum.second,-sum.first);
	final_vel.mag = 1;
}

AliceStructs::pnt Rules::charge()
{
	float dx;
	float dy;

	float min_sep = 1000.0;
	float check_sep; //sep distance

	for (int i=0; i < model.rel_chargers.size(); i++)
	{
		if((!model.abs_chargers->at(i).occupied) && (!model.committed)) //charger is open or charger not committed
		{
			dx = model.rel_chargers.at(i).x;
			dy = model.rel_chargers.at(i).y;
			check_sep = sqrt(pow(dx,2) + pow(dy,2)); //check separation distance

			std::cout<<"charger: "<<i<<":"<<"dx,dy: "<<dx<<","<<dy<<"|"<<check_sep<<std::endl;

			if(check_sep < min_sep)
			{
				model.closest_pos = i; //saves pos of closest
				min_sep = check_sep; //updates min_sep
			}
		}
		std::cout<<"why you break? "<<i<<std::endl;
	}
	AliceStructs::pnt go_to;
	std::cout<<model.closest_pos<<" :closest pos"<<std::endl;
	go_to.x = model.rel_chargers.at(model.closest_pos).x;
	go_to.y = model.rel_chargers.at(model.closest_pos).y;

	if(model.battery_lvl<0.1) //assign priority
	{
		go_to.z = 2; //given highest priority
		model.abs_chargers->at(model.closest_pos).occupied = true; //charger is "checked out".
		model.committed = true;
		model.rel_chargers.at(model.closest_pos).occupied = true;
	}else
	{
		go_to.z = 0;
	}

//	std::pair<float,float> temp_pnt = model.transformCur(go_to.x, go_to.y); //transform into first frame of bot (reasons).
//	go_to.x = temp_pnt.first;
//	go_to.y = temp_pnt.second;

	return go_to;
}

//--------------------Still need implementations-------------------------------------------
bool Rules::checkCollisions()
{

}

void Rules::rest() //gives a command to not move.
{
	//std::pair<float,float> do_not = model.transformCur(0,0); //puts null command into first frame (reasons).
//	model.goTo.x = do_not.first;
//	model.goTo.y = do_not.second;

		model.goTo.x = 0;
		model.goTo.y = 0;
}
//-----------------------------------------------------------------------------------------
//bool Rules::changeState()
//{ //finds highest priority in list and coorespoinding state.
//	bool result;
//	float highest_prior = 0;
//	int highest_i;
//	for(int i = 0; i < UNUSED; i ++)
//	{
//		if(model.priority->at(i) > highest_prior)
//		{
//			highest_prior = model.priority->at(i);
//			highest_i = i;
//		}
//	}
//
//#if DEBUG_schange
//	std::cout<<"==========Pre adjustment===========\n";
//	std::cout<<"state: "<<(int)state<<std::endl;
//	std::cout<<"highest_i: "<<highest_i<<std::endl;
//	std::cout<<"priority:  "<<highest_prior<<std::endl;
//	std::cout<<"=====================\n";
//#endif
//
//	if(highest_i != (int)state) //if a different state has higher priority
//	{
//		result = true;
//		state = (State)highest_i; //this has been verified to produce the correct output.
//
//#if DEBUG_schange
//		std::cout<<"==========Post adjustment===========\n";
//		std::cout<<"state: "<<(int)state<<std::endl;
//		std::cout<<"highest_i: "<<highest_i<<std::endl;
//		std::cout<<"priority:  "<<highest_prior<<std::endl;
//		std::cout<<"=====================\n";
//#endif
//
//	}
//	else
//	{
//		result = false;
//	}
//	return result;
//}
//
//bool Rules::updateWaypoint() //checks if action should be taken (if near goTo or new rule has higher priority).
//{//priority received in order {REST, CHARGE, CONTOUR, TARGET, EXPLORE, UNUSED}.
//	std::cout << model.goTo.x << model.goTo.y << std::endl;
//	bool changed; //tells you if priorities of rules have changed.
//	bool take_action = false; //only made true if near waypoint, or if state priorities change.
//
//	float tolerance = 1; //arbitrary limit of how close the bot needs to get to a waypoint for it to count as reaching it.
//	std::pair<float,float> waypoint = model.transformFir(model.goTo.x,model.goTo.y);
//	float r = sqrt(pow(waypoint.first - model.cur_pose.x,2) + pow(waypoint.second - model.cur_pose.y,2));
//
//	if(r<tolerance) //if inside
//	{
//		changed = changeState();
//		take_action = true;
//		std::cout<<(changed? "true" : "false")<<std::endl;
//	}
//	else if(changeState())
//	{
//		changed = true;
//		take_action = true;
//	}
//	if(take_action)
//	{
//		switch((int)state)
//		{
//			case 1: charge();
//			case 2:	findContour();
//			case 3: goToTar();
//			case 4: explore();
//			default: rest();
//		}
//	}
//	return take_action;
//}

void Rules::updateVel(AliceStructs::vel *_fv) //puts the final_velocity in the frame of the bot.
{
	std::pair<float,float> cur_goTo = model.transformFir(model.goTo.x, model.goTo.y);
	float magnitude = sqrt(pow(cur_goTo.first,2) + pow(cur_goTo.second,2));
	float direction = atan2(cur_goTo.second,cur_goTo.first);
	_fv->mag = magnitude;
	_fv->dir = direction;
}


AliceStructs::pnt Rules::goToTar()
{
	float temp = 10000;
	AliceStructs::pnt to_return;
	to_return.z = 0;
	for (auto& tar : model.targets)
	{
		float check = calcDis(tar.x, tar.y, 0, 0);
		if (check < temp)
		{
			temp = check;
			to_return = tar;
			to_return.z = 1;
		}
	}
	return to_return;
}

void Rules::findContour()
{
	AliceStructs::pose best;
	float pri = 0; //uses a formula based on distance, recency(confidence), and strength
	std::pair<float, float> temp = model.transformFtF(model.cur_pose.x, model.cur_pose.y, 0, 0, 0); //transforms the cur_pose to the first_pose

	for (auto& contour : model.archived_contour)
	{
		float temp_pri = (contour.z-model.cur_pose.z) / (10 + model.time.sec - contour.time.sec)
				/ pow(calcDis(temp.first, temp.second, contour.x, contour.y), 0.5);

		if (temp_pri > pri)
		{
			best = contour;
			pri = temp_pri;
		}
	}
	final_vel.dir = atan2(best.y - temp.second, best.x - temp.first) + model.first_pose.heading - model.cur_pose.heading;
	final_vel.mag = 1;
	model.goTo.x = best.x;
	model.goTo.y = best.y;
}

float Rules::calcDis(float _x1, float _y1, float _x2, float _y2)
{
	return pow(pow(_x1 - _x2, 2) + pow(_y1 - _y2, 2), 0.5);
}

std::pair<float, float> Rules::calcQuad(float a, float b, float c)
{
	std::pair<float, float> to_return;
	to_return.first = (-b + pow(pow(b, 2) - 4*a*c, 0.5))/(2*a);
	to_return.second = (-b - pow(pow(b, 2) - 4*a*c, 0.5))/(2*a);
	return to_return;
}

bool Rules::checkBlocked()
{
	for (auto& zone : findDeadZones())
	{
		if (zone.first.first < atan2(model.goTo.y, model.goTo.x) < zone.first.second
				|| zone.first.first > atan2(model.goTo.y, model.goTo.x) > zone.first.second)
		{
			return true;
		}
	}
	return false;
}

//std::string Rules::checkBattery(std::string state)
//{
//	float acceptable_lvl = model.battery_lvl * 0.2;
//	if(model.battery_lvl < acceptable_lvl && state != "charging")
//	{
//		state = "needs_charging";
//	}
//	return state;
//}
AliceStructs::pnt Rules::findPath(AliceStructs::pnt waypnt, std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones)
{
	float tf = atan2(waypnt.y, waypnt.x);
	AliceStructs::pnt to_return;
	to_return.x = model.goTo.x;
	to_return.y = model.goTo.y;
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> blockers;
	std::cout << "finding path" << std::endl;
	for (auto& zone : dead_zones)
	{
		//std::cout << "found a zone" << std::endl;
		if (pow(pow(zone.second.x_off, 2) + pow(zone.second.y_off, 2), 0.5) < pow(pow(waypnt.x, 2) + pow(waypnt.y, 2), 0.5))
		{
			//std::cout << "it's on the proper side of me" << std::endl;
			//std::cout << fabs(tf) << " first " << fabs(zone.first.first) << " second " << fabs(zone.first.second) << std::endl;
			if (((zone.first.first < tf && tf < zone.first.second) || (zone.first.second < tf && tf < zone.first.first))
				|| ((fabs(zone.first.first - zone.first.second) > M_PI/2) &&
						((fabs(tf) + M_PI/12 > fabs(zone.first.first)) && (fabs(tf) + M_PI/12 > fabs(zone.first.second)))))
			{
				std::cout << "it's in my way" << std::endl;
				blockers.push_back(zone);
			}
		}
	}
	std::pair<std::pair<float, AliceStructs::obj>, std::pair<float, AliceStructs::obj>> angles;
	angles.first.first = 0;
	angles.second.first = 0;
	for (auto& blocker : blockers)
	{
		blocker.first.first -= tf;
		blocker.first.second -= tf;
		//voltron(blocker)
		if (blocker.first.first > blocker.first.second)
		{
			std::cout << "First is greater" << std::endl;
			if (blocker.first.first > angles.first.first)
			{
				angles.first.first = blocker.first.first;
				angles.first.second = blocker.second;
			}
			if (blocker.first.second < angles.second.first)
			{
				angles.second.first = blocker.first.second;
				angles.second.second = blocker.second;
			}
		}
		else
		{
			std::cout << "Second is greater" << std::endl;
			if (blocker.first.second > angles.first.first)
			{
				angles.second.first = blocker.first.second;
				angles.second.second = blocker.second;
			}
			if (blocker.first.first < angles.second.first)
			{
				angles.first.first = blocker.first.first;
				angles.first.second = blocker.second;
			}
		}
	}
	std::pair<float, AliceStructs::obj> final;
	angles.first.first += tf;
	angles.second.first += tf;
	std::cout << "first " << angles.first.first << " second " << angles.second.first << " tf " << tf << std::endl;
	if (fabs(fabs(angles.first.first) - fabs(tf)) < fabs(fabs(angles.second.first) - fabs(tf)))
	{
		final = angles.first;
	}
	else
	{
		final = angles.second;
	}
	std::cout << final.first << std::endl;
	std::cout << "x: " << final.second.x_off << " y: " << final.second.y_off << std::endl;
	std::cout << "q: " << final.second.x_rad << " p: " << final.second.y_rad << std::endl;
	if (final.first != tf)
	{
		float m = tan(final.first);
		std::cout << "m: " << m << std::endl;
		to_return.x = 0.5*(2*final.second.x_off/pow(final.second.x_rad, 2) + 2*final.second.y_off*m/pow(final.second.y_rad, 2))/
				(1/pow(final.second.x_rad, 2) + pow(m, 2)/pow(final.second.y_rad, 2));
		to_return.y = 0.5*(2*final.second.y_off/pow(final.second.y_rad, 2) + 2*final.second.x_off/(m*pow(final.second.x_rad, 2)))/
				(1/pow(final.second.y_rad, 2) + 1/(pow(m, 2) * pow(final.second.x_rad, 2)));
		std::cout << "test: " << (2*final.second.y_off*m) << std::endl;
		std::cout << "x: " << to_return.x << " y: " << to_return.y << std::endl;
		std::cout << atan2(to_return.y, to_return.x) << std::endl;
	}
	else
	{
		to_return = model.goTo;
	}
	return to_return;
}

std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> Rules::findDeadZones()
{
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones;
	for (auto& obs : model.obstacles)
	{
		//float obs.x_off = model.cur_pose.x - obs.x_off;
		//float obs.y_off = model.cur_pose.y - obs.y_off;
		std::pair<float, float> aoe = calcQuad(pow(obs.x_off, 2) - pow(obs.x_rad + model.SIZE + model.SAFE_DIS, 2), -2 * obs.x_off * obs.y_off, pow(obs.y_off, 2) - pow(obs.y_rad + model.SIZE + model.SAFE_DIS, 2));
		//Unfortunately, trig can't handle quadrents very well, so we have to do it ourselves
		if (obs.y_off - obs.y_rad - model.SIZE - model.SAFE_DIS < 0)
		{
			if (obs.x_off + obs.x_rad + model.SIZE + model.SAFE_DIS < 0)
			{
				//std::cout << "top right" << std::endl;
				aoe.first = -M_PI+atan(aoe.first);
			}
			else
			{
				//std::cout << "top left" << std::endl;
				aoe.first = atan(aoe.first);
			}
		}
		else
		{
			if (obs.x_off - obs.x_rad - model.SIZE - model.SAFE_DIS< 0)
			{
				//std::cout << "bottom right" << std::endl;
				aoe.first = M_PI+atan(aoe.first);
			}
			else
			{
				//std::cout << "bottom left" << std::endl;
				aoe.first = atan(aoe.first);
			}
		}
		if (obs.y_off + obs.y_rad + model.SIZE + model.SAFE_DIS < 0)
		{
			if (obs.x_off - obs.x_rad - model.SIZE - model.SAFE_DIS < 0)
			{
				//std::cout << "top right" << std::endl;
				aoe.second = -M_PI + atan(aoe.second);
			}
			else
			{
				//std::cout << "top left" << std::endl;
				aoe.second = atan(aoe.second);
			}
		}
		else
		{
			if (obs.x_off + obs.x_rad + model.SIZE + model.SAFE_DIS < 0)
			{
				//std::cout << "bottom right" << std::endl;
				aoe.second = M_PI + atan(aoe.second);
			}
			else
			{
				//std::cout << "bottom left" << std::endl;
				aoe.second = atan(aoe.second);
			}
		}
		std::cout << "dead zones: " << aoe.first << " - " << aoe.second << std::endl;
		std::pair<std::pair<float, float>, AliceStructs::obj> to_push;
		to_push.first = aoe;
		to_push.second = obs;
		dead_zones.push_back(to_push);
	}
	return dead_zones;
}

bool Rules::avoidNeighbors()
{
	bool to_return = false;
	for (auto& bot : model.neighbors)
	{
		float x_int = ((bot.y - bot.tar_y)/(bot.x - bot.tar_x)*bot.x - (model.cur_pose.y - model.goTo.y)/(model.cur_pose.x - model.goTo.y)*model.cur_pose.x - bot.y + model.cur_pose.y)/
				((bot.y - bot.tar_y)/(bot.x - bot.tar_x) - (model.cur_pose.y - model.goTo.y)/(model.cur_pose.x - model.goTo.x));
		float y_int = ((bot.x - bot.tar_x)/(bot.y - bot.tar_y)*bot.y - (model.cur_pose.x - model.goTo.x)/(model.cur_pose.y - model.goTo.y)*model.cur_pose.y + model.cur_pose.x - bot.x)/
				((bot.x - bot.tar_x)/(bot.y - bot.tar_y) - (model.cur_pose.x - model.goTo.x)/(model.cur_pose.y - model.goTo.y));
		float self_tti = (model.cur_pose.heading - atan2(model.goTo.y, model.goTo.x)/model.MAX_AV +
				calcDis(x_int, y_int, model.cur_pose.x, model.cur_pose.y)/model.MAX_LV);
		float bot_tti = (bot.ang - atan2(bot.tar_y, bot.tar_x)/model.MAX_AV +
				calcDis(x_int, y_int, bot.x, bot.y)/model.MAX_LV);
		if (abs(self_tti - bot_tti) < checkTiming(x_int, y_int, bot)/model.MAX_LV)
		{
			std::pair<float, float> slopes = calcQuad(pow(model.cur_pose.x - x_int, 2) - pow(margin, 2),
					2*(model.cur_pose.x - x_int)*(model.cur_pose.y - y_int),
					pow(model.cur_pose.y - y_int, 2) - pow(margin, 2));
			std::pair<float, float> new_tar_1;
			new_tar_1.first = (slopes.first*model.cur_pose.x - model.cur_pose.y + x_int/slopes.first + y_int)/(1/slopes.first + slopes.first);
			new_tar_1.second = (slopes.first*y_int + x_int + model.cur_pose.y/slopes.first - model.cur_pose.x)/(1/slopes.first + slopes.first);
			std::pair<float, float> new_tar_2;
			new_tar_2.first = (slopes.second*model.cur_pose.x - model.cur_pose.y + x_int/slopes.second + y_int)/(1/slopes.second + slopes.second);
			new_tar_2.second = (slopes.second*y_int + x_int + model.cur_pose.y/slopes.second - model.cur_pose.x)/(1/slopes.second + slopes.second);
			std::pair<float, float> new_tar;
			if (calcDis(new_tar_1.first, new_tar_1.second, bot.x, bot.y) < calcDis(new_tar_2.first, new_tar_2.second, bot.x, bot.y))
			{
				new_tar = new_tar_1;
			}
			else
			{
				new_tar = new_tar_2;
			}
			if (calcDis(new_tar.first, new_tar.second, model.cur_pose.x, model.cur_pose.y) < calcDis(model.goTo.x, model.goTo.y, model.cur_pose.x, model.cur_pose.y))
			{
				model.goTo.x = new_tar.first;
				model.goTo.y = new_tar.second;
				to_return = true;
			}
		}
	}
	return to_return;
}

float Rules::checkTiming(float _x_int, float _y_int, AliceStructs::neighbor bot)
{
	float tcpx = -pow(pow(margin, 2)/(pow((bot.tar_x - bot.x)/(bot.y - bot.tar_y), 2) + 1), 0.5) + _x_int;
	float tcpy = -pow(pow(margin, 2)/(pow((bot.y - bot.tar_y)/(bot.tar_x - bot.x), 2) + 1), 0.5) + _y_int;
	float adj_x_int = (-(model.cur_pose.y - model.goTo.y)/(model.cur_pose.x - model.goTo.x)*model.cur_pose.x + model.cur_pose.y + (bot.y - bot.tar_y)/(bot.x - bot.tar_x)*tcpx - tcpy)/
			((bot.y - bot.tar_y)/(bot.x - bot.tar_x) - (model.cur_pose.y - model.goTo.y)/(model.cur_pose.x - model.goTo.x));
	float adj_y_int = (-(bot.x - bot.tar_x)/(bot.y - bot.tar_y)*tcpy + tcpx + (model.cur_pose.x - model.goTo.x)/(model.cur_pose.y - model.goTo.y)*model.cur_pose.y - model.cur_pose.x)/
			((model.cur_pose.x - model.goTo.x)/(model.cur_pose.y - model.goTo.y) - (bot.x - bot.tar_x)/(bot.y - bot.tar_y));
	if ((bot.y - bot.tar_y)/(bot.x - bot.tar_x) > 0)
	{
		return -calcDis(adj_x_int, adj_y_int, _x_int, _y_int);
	}
	else
	{
		return calcDis(adj_x_int, adj_y_int, _x_int, _y_int);
	}
}

void Rules::checkForProblems()
{
	if(model.priority->size() != (int)UNUSED)
	{
		std::cout<<"--------------ERROR: priority vector from <hawk_sim> does not have "
				"the priorities for each state in <enum State>------------------"<<std::endl;
	}
}
