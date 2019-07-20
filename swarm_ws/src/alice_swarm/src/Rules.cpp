#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include <map>
#include <string>
#include <math.h>
#include <iostream>
#include <bits/stdc++.h>

#define DEBUG_schange 1
#define DEBUG_chargeing 0

Rules::Rules()
{
	state = REST;
}

Rules::Rules(Model &_model) :
		model(&_model)
{
	state = REST;
}

void Rules::init(Model &_model)
{
	_model.goTo.x = 0;
	_model.goTo.y = 0;
	first_time = false;
}
//===================================================================================================================

void Rules::stateLoop(Model &_model)
{
	if (first_time)
		init(_model); //need this to solve the ghost of random goto initialization
	model = &_model;
	std::pair<float, float> cur_temp = _model.transformFir(_model.goTo.x, _model.goTo.y); //Shifts the frame of goTo to the current frame for calculations
	cur_go_to.x = cur_temp.first;
	cur_go_to.y = cur_temp.second;
	bool run = shouldLoop();
	if (run)
	{
		std::vector<AliceStructs::pnt> go_to_list;

		// add other rules here
			go_to_list.push_back(findContour());
// go_to_list.push_back(goToTar());
//	go_to_list.push_back(charge());
//		go_to_list.push_back(rest());
	//	go_to_list.push_back(explore());

		float temp = -1;
		for (auto &rule : go_to_list)
		{
//			std::cout<<"the rule z is: "<<rule.z<<std::endl;
			if (rule.z > temp)
			{
				temp = rule.z;

				cur_go_to.x = rule.x;
				cur_go_to.y = rule.y;

//				std::cout << "theta: "<<atan2(model->goTo.y, model->goTo.x) << std::endl;
//				std::cout<<"(x,y): "<<model->goTo.x<<","<<model->goTo.y<<std::endl;
			}
			//}
			//	avoidCollisions();
//			float mag = sqrt(pow(model->goTo.x,2) + pow(model->goTo.y,2));
//			std::cout<<"(x,y): "<<model->goTo.x<<","<<model->goTo.y<<"| "<<mag<<std::endl;
//			std::cout<<"-----------------------------------"<<std::endl;
		}

		_model.goTo.time = ros::Time::now(); //time stamps when the goto is created
	}
	goToTimeout();
	std::pair<float, float> back_to_fir = _model.transformCur(cur_go_to.x, cur_go_to.y);
	_model.goTo.x = back_to_fir.first;
	_model.goTo.y = back_to_fir.second;
}

//===================================================================================================================
void Rules::goToTimeout()
{
	if (ros::Time::now().sec - model->goTo.time.sec > 20) //if it's been more than 20 seconds, move goTo
	{

		float mag = calcDis(cur_go_to.x, cur_go_to.y, 0, 0);

		//for handling bad goTo points (too far)
		cur_go_to.x *= 30 / mag;
		cur_go_to.y *= 30 / mag;

		cur_go_to.x *= -1; //flip sign for handling unreachable goTo's
		cur_go_to.y *= -1;
		model->goTo.time = ros::Time::now(); //time stamps when the goto is changed
	}

}

bool Rules::shouldLoop()
{
	bool result = false;
	if (calcDis(cur_go_to.x, cur_go_to.y, 0, 0) < model->SIZE / 2) //if the bot has made it to the way point
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
	return result;
}

//===================================================================================================================

void Rules::avoidCollisions()
{
	bool checker = true;
	//while (checker)
	//{
	model->goTo = findPath(model->goTo, findDeadZones());
	//checker = avoidNeighbors();
	//}
}

//TO USE ANY OF THESE RULES, USE THE MODEL'S POINTER AND CHANGE SYNTAX ACCORDINGLY

AliceStructs::pnt Rules::explore()
{
	AliceStructs::pose best;
	std::pair<float, float> sum(0, 0);
	for (auto &contour : model->archived_contour)
	{
		std::pair<float, float> temp = model->transformFir(contour.x, contour.y); //transforms to the current frame
		float dist = calcDis(0, 0, contour.x, contour.y);
		if (dist < model->vision) //simplistic, just adds the vectors together and goes the opposite way
		{
			sum.first += temp.first;
			sum.second += temp.second; //some form of priority assigned by confidence TBD
		}
	}
	float mag = calcDis(sum.first, sum.second, 0, 0);
	if (mag > 1)
	{
		sum.first *= 10 / mag;
		sum.second *= 10 / mag;
	}
	AliceStructs::pnt to_return;
	to_return.x = -sum.first;
	to_return.y = -sum.second;
	to_return.z = 3;
	return to_return;

}

AliceStructs::pnt Rules::charge()
{
	AliceStructs::pnt go_to;
	float dx;
	float dy;
	float min_sep = model->min_sep;
	float check_sep; //sep distance
	bool odd_man_out = false;
	if (!availableChargers() && !model->committed) //makes this a non op when no chargers available.
	{
		odd_man_out = true;
	}

	if (model->abs_chargers->size() > 0 && !odd_man_out) //and available chargers
	{
		if (!model->charge2) //runs code for first stage if 2nd not initiated.
		{
			for (int i = 0; i < model->rel_chargers.size(); i++) //checks each charger
			{
				if ((!model->abs_chargers->at(i).occupied) && (!model->committed)) //charger is open or charger not committed
				{
					dx = model->rel_chargers.at(i).target_x;
					dy = model->rel_chargers.at(i).target_y;
					check_sep = sqrt(pow(dx, 2) + pow(dy, 2)); //check separation distance
					std::cout << "charger: " << i << ":" << "dx,dy: " << dx << "," << dy << "|" << check_sep << std::endl;
					if (check_sep < min_sep) //if the closest
					{
						model->closest_pos = i; //saves pos of closest
						min_sep = check_sep; //updates min_sep
						model->min_sep = min_sep;
					}
				}
				std::cout << "why you break? " << i << std::endl;
			}
			std::cout << model->closest_pos << " :closest pos" << std::endl;
			go_to.x = model->rel_chargers.at(model->closest_pos).target_x; //sets pos of closest charger as target.
			go_to.y = model->rel_chargers.at(model->closest_pos).target_y;

			if ((pow(go_to.x, 2) + pow(go_to.y, 2)) < pow(model->SIZE/5, 2) && (model->committed)) //initiates charge2 after it reaches first waypoint.
			{
				model->charge2 = true;
			}
			std::cout << "the battery level is: " << model->battery_lvl << std::endl;
			if ((model->battery_lvl < 3.8)) //assign priority
			{
				go_to.z = 2; //given highest priority
				model->abs_chargers->at(model->closest_pos).occupied = true; //charger is "checked out".
				model->committed = true;
				model->rel_chargers.at(model->closest_pos).occupied = true;
			} else
			{
				std::cout << "saftey is running" << std::endl;
				//makes sure that the reset sticks. There is lag in reseting these vars sometimes.
				model->abs_chargers->at(model->closest_pos).occupied = false;
				model->rel_chargers.at(model->closest_pos).occupied = false;
				model->committed = false;
				go_to.z = 0;
			}
		} else if (model->charge2) //runs code for second waypoint.
		{
#if DEBUG_chargeing
//			std::cout<<"++++++++++++++++++++++++++++++\n";
//			std::cout<<"charge2 activated"<<std::endl;
//			std::cout<<(model->abs_chargers->at(model->closest_pos).occupied ? "true" : "false")<<std::endl;
//			std::cout<<"charged()"<<(charged() ? "true" : "false")<<std::endl;
#endif
			go_to = charge2();

			if (charged()) //resets charging vars, and makes priority of the generated point zero.
			{
#if DEBUG_chargeing
				std::cout<<"************************************************************\n";
				std::cout<<"HARD RESET TO CHARGING"<<std::endl;
				if((!model->charging) && (!model->committed) && (!model->charge2) &&
				(!model->abs_chargers->at(model->closest_pos).occupied) && (!model->rel_chargers.at(model->closest_pos).occupied))
				{
					std::cout<<"Reset successful"<<std::endl;
				}
				std::cout<<"************************************************************\n";
#endif
				go_to.z = 0;
			}
		}
	} else
	{
//		std::cout<<"FUCK MY LIFE\n";
		go_to.x = 0;
		go_to.y = 0;
		go_to.z = 0;
	}
	return go_to;
}
AliceStructs::pnt Rules::charge2() //phase 2 of the charging sequence.
{
	AliceStructs::pnt go_to;
//	std::cout<<"the battery level is: "<<model->battery_lvl<<std::endl;
	if (model->battery_state != CHARGING)
	{
		go_to.x = model->rel_chargers.at(model->closest_pos).x;
		go_to.y = model->rel_chargers.at(model->closest_pos).y;
		model->min_sep = sqrt(pow(go_to.x, 2) + pow(go_to.y, 2));
//		std::cout<<"charge2 x,y: "<<go_to.x<<","<<go_to.y<<std::endl;
//		std::cout<<"++++++++++++++++++++++++++++++\n";
	} else
	{
		go_to = rest();
	}
	go_to.z = 3;
	return go_to;
}
bool Rules::charged()
{
	bool return_bool = false;
//	std::cout<<"the battery level is: "<<model->battery_lvl<<std::endl;
//	std::cout<<"the battery state is: "<<(int)model.battery_state <<std::endl;
//	if(model.battery_lvl > 4.8)
	if (model->battery_state == CHARGED)
	{
		return_bool = true;
		model->charging = false;
		model->committed = false;
		model->charge2 = false;
		model->abs_chargers->at(model->closest_pos).occupied = false;
		model->rel_chargers.at(model->closest_pos).occupied = false;
		model->min_sep = 1000.0;
	}
	return return_bool;
}
bool Rules::availableChargers()
{
	bool result = false;
	int i = 0;
	while (i < model->rel_chargers.size())
	{
		if (!model->rel_chargers.at(i).occupied)
		{
			result = true;
			i = model->rel_chargers.size() + 1;
		}
		i++;
	}
	return result;
}
//--------------------Still need implementations-------------------------------------------
bool Rules::checkCollisions()
{
	cur_go_to = findPath(cur_go_to, findDeadZones());
}

AliceStructs::pnt Rules::rest() //gives a command to not move.
{
	//returns its' own location with a set priority
	AliceStructs::pnt go_to;
	go_to.x = 0;
	go_to.y = 0;
	go_to.z = 3;
	return go_to;
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
AliceStructs::pnt Rules::goToTar()
{
	float temp = 10000;
	AliceStructs::pnt to_return;
	to_return.x = 0;
	to_return.y = 0;
	to_return.z = 0;
	for (auto &tar : model->targets)
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

float Rules::calcDis(float _x1, float _y1, float _x2, float _y2)
{
	return pow(pow(_x1 - _x2, 2) + pow(_y1 - _y2, 2), 0.5);
}

std::pair<float, float> Rules::calcQuad(float a, float b, float c)
{
	std::pair<float, float> to_return;
	to_return.first = (-b + pow(pow(b, 2) - 4 * a * c, 0.5)) / (2 * a);
	to_return.second = (-b - pow(pow(b, 2) - 4 * a * c, 0.5)) / (2 * a);
	return to_return;
}

AliceStructs::pnt Rules::findContour()
{
	AliceStructs::pose best;
	AliceStructs::pnt to_return;
	to_return.x = 0;
	to_return.y = 0;
	to_return.z = 0;
	float pri = 0; //uses a formula based on distance, recency(confidence), and strength
	std::pair<float, float> temp = model->transformCur(0, 0); //transforms the cur_pose to the first_pose
	bool init = false;
	for (auto &contour : model->archived_contour)
	{
		init = true;
		float temp_pri = (contour.z); // - model->cur_pose.z) / (10 + model->time.sec - contour.time.sec)
		/// pow(calcDis(temp.first, temp.second, contour.x, contour.y), 0.5);

		if (temp_pri > pri)
		{
			best = contour;
			pri = temp_pri;
		}
	}
	if (init)
	{
		std::pair<float, float> go = model->transformFir(best.x, best.y);
		to_return.x = go.first;
		to_return.y = go.second;
		to_return.z = pri;
	}
	return to_return;
}

/*
 bool Rules::checkBlocked()
 {
 for (auto &zone : findDeadZones())
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
 */
AliceStructs::pnt Rules::findPath(AliceStructs::pnt waypnt,
		std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones)
{
	float tf = atan2(waypnt.y, waypnt.x);
	AliceStructs::pnt to_return;
	to_return.x = model->goTo.x;
	to_return.y = model->goTo.y;
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> blockers;
	std::cout << "finding path" << std::endl;
	for (auto &zone : dead_zones)
	{
		//std::cout << "found a zone" << std::endl;
		if (pow(pow(zone.second.x_off, 2) + pow(zone.second.y_off, 2), 0.5) < pow(pow(waypnt.x, 2) + pow(waypnt.y, 2), 0.5))
		{
			//std::cout << "it's on the proper side of me" << std::endl;
			//std::cout << fabs(tf) << " first " << fabs(zone.first.first) << " second " << fabs(zone.first.second) << std::endl;
			if (((zone.first.first < tf && tf < zone.first.second) || (zone.first.second < tf && tf < zone.first.first))
					|| ((fabs(zone.first.first - zone.first.second) > M_PI / 2)
							&& ((fabs(tf) + M_PI / 12 > fabs(zone.first.first)) && (fabs(tf) + M_PI / 12 > fabs(zone.first.second)))))
			{
				std::cout << "it's in my way" << std::endl;
				blockers.push_back(zone);
			}
		}
	}
	std::pair<std::pair<float, AliceStructs::obj>, std::pair<float, AliceStructs::obj>> angles;
	angles.first.first = 0;
	angles.second.first = 0;
	for (auto &blocker : blockers)
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
		} else
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
	} else
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
		to_return.x =
				0.5
						* (2 * final.second.x_off / pow(final.second.x_rad, 2)
								+ 2 * final.second.y_off * m / pow(final.second.y_rad, 2))
						/ (1 / pow(final.second.x_rad, 2) + pow(m, 2) / pow(final.second.y_rad, 2));
		to_return.y = 0.5
				* (2 * final.second.y_off / pow(final.second.y_rad, 2)
						+ 2 * final.second.x_off / (m * pow(final.second.x_rad, 2)))
				/ (1 / pow(final.second.y_rad, 2) + 1 / (pow(m, 2) * pow(final.second.x_rad, 2)));
		std::cout << "test: " << (2 * final.second.y_off * m) << std::endl;
		std::cout << "x: " << to_return.x << " y: " << to_return.y << std::endl;
		std::cout << atan2(to_return.y, to_return.x) << std::endl;
	} else
	{
		to_return = model->goTo;
	}
	return to_return;
}

std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> Rules::findDeadZones()
{
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones;
	for (auto &obs : model->obstacles)
	{
		//float obs.x_off = model->cur_pose.x - obs.x_off;
		//float obs.y_off = model->cur_pose.y - obs.y_off;
		std::pair<float, float> aoe = calcQuad(pow(obs.x_off, 2) - pow(obs.x_rad + model->SIZE + model->SAFE_DIS, 2),
				-2 * obs.x_off * obs.y_off, pow(obs.y_off, 2) - pow(obs.y_rad + model->SIZE + model->SAFE_DIS, 2));
		//Unfortunately, trig can't handle quadrents very well, so we have to do it ourselves
		if (obs.y_off - obs.y_rad - model->SIZE - model->SAFE_DIS < 0)
		{
			if (obs.x_off + obs.x_rad + model->SIZE + model->SAFE_DIS < 0)
			{
				//std::cout << "top right" << std::endl;
				aoe.first = -M_PI + atan(aoe.first);
			} else
			{
				//std::cout << "top left" << std::endl;
				aoe.first = atan(aoe.first);
			}
		} else
		{
			if (obs.x_off - obs.x_rad - model->SIZE - model->SAFE_DIS < 0)
			{
				//std::cout << "bottom right" << std::endl;
				aoe.first = M_PI + atan(aoe.first);
			} else
			{
				//std::cout << "bottom left" << std::endl;
				aoe.first = atan(aoe.first);
			}
		}
		if (obs.y_off + obs.y_rad + model->SIZE + model->SAFE_DIS < 0)
		{
			if (obs.x_off - obs.x_rad - model->SIZE - model->SAFE_DIS < 0)
			{
				//std::cout << "top right" << std::endl;
				aoe.second = -M_PI + atan(aoe.second);
			} else
			{
				//std::cout << "top left" << std::endl;
				aoe.second = atan(aoe.second);
			}
		} else
		{
			if (obs.x_off + obs.x_rad + model->SIZE + model->SAFE_DIS < 0)
			{
				//std::cout << "bottom right" << std::endl;
				aoe.second = M_PI + atan(aoe.second);
			} else
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
	for (auto &bot : model->neighbors)
	{
		float x_int = ((bot.y - bot.tar_y) / (bot.x - bot.tar_x) * bot.x
				- (model->cur_pose.y - model->goTo.y) / (model->cur_pose.x - model->goTo.y) * model->cur_pose.x - bot.y
				+ model->cur_pose.y)
				/ ((bot.y - bot.tar_y) / (bot.x - bot.tar_x)
						- (model->cur_pose.y - model->goTo.y) / (model->cur_pose.x - model->goTo.x));
		float y_int = ((bot.x - bot.tar_x) / (bot.y - bot.tar_y) * bot.y
				- (model->cur_pose.x - model->goTo.x) / (model->cur_pose.y - model->goTo.y) * model->cur_pose.y
				+ model->cur_pose.x - bot.x)
				/ ((bot.x - bot.tar_x) / (bot.y - bot.tar_y)
						- (model->cur_pose.x - model->goTo.x) / (model->cur_pose.y - model->goTo.y));
		float self_tti = (model->cur_pose.heading - atan2(model->goTo.y, model->goTo.x) / model->MAX_AV
				+ calcDis(x_int, y_int, model->cur_pose.x, model->cur_pose.y) / model->MAX_LV);
		float bot_tti = (bot.ang - atan2(bot.tar_y, bot.tar_x) / model->MAX_AV
				+ calcDis(x_int, y_int, bot.x, bot.y) / model->MAX_LV);
		if (abs(self_tti - bot_tti) < checkTiming(x_int, y_int, bot) / model->MAX_LV)
		{
			std::pair<float, float> slopes = calcQuad(pow(model->cur_pose.x - x_int, 2) - pow(margin, 2),
					2 * (model->cur_pose.x - x_int) * (model->cur_pose.y - y_int),
					pow(model->cur_pose.y - y_int, 2) - pow(margin, 2));
			std::pair<float, float> new_tar_1;
			new_tar_1.first = (slopes.first * model->cur_pose.x - model->cur_pose.y + x_int / slopes.first + y_int)
					/ (1 / slopes.first + slopes.first);
			new_tar_1.second = (slopes.first * y_int + x_int + model->cur_pose.y / slopes.first - model->cur_pose.x)
					/ (1 / slopes.first + slopes.first);
			std::pair<float, float> new_tar_2;
			new_tar_2.first = (slopes.second * model->cur_pose.x - model->cur_pose.y + x_int / slopes.second + y_int)
					/ (1 / slopes.second + slopes.second);
			new_tar_2.second = (slopes.second * y_int + x_int + model->cur_pose.y / slopes.second - model->cur_pose.x)
					/ (1 / slopes.second + slopes.second);
			std::pair<float, float> new_tar;
			if (calcDis(new_tar_1.first, new_tar_1.second, bot.x, bot.y)
					< calcDis(new_tar_2.first, new_tar_2.second, bot.x, bot.y))
			{
				new_tar = new_tar_1;
			} else
			{
				new_tar = new_tar_2;
			}
			if (calcDis(new_tar.first, new_tar.second, model->cur_pose.x, model->cur_pose.y)
					< calcDis(model->goTo.x, model->goTo.y, model->cur_pose.x, model->cur_pose.y))
			{
				model->goTo.x = new_tar.first;
				model->goTo.y = new_tar.second;
				to_return = true;
			}
		}
	}
	return to_return;
}

float Rules::checkTiming(float _x_int, float _y_int, AliceStructs::neighbor bot)
{
	float tcpx = -pow(pow(margin, 2) / (pow((bot.tar_x - bot.x) / (bot.y - bot.tar_y), 2) + 1), 0.5) + _x_int;
	float tcpy = -pow(pow(margin, 2) / (pow((bot.y - bot.tar_y) / (bot.tar_x - bot.x), 2) + 1), 0.5) + _y_int;
	float adj_x_int = (-(model->cur_pose.y - model->goTo.y) / (model->cur_pose.x - model->goTo.x) * model->cur_pose.x
			+ model->cur_pose.y + (bot.y - bot.tar_y) / (bot.x - bot.tar_x) * tcpx - tcpy)
			/ ((bot.y - bot.tar_y) / (bot.x - bot.tar_x)
					- (model->cur_pose.y - model->goTo.y) / (model->cur_pose.x - model->goTo.x));
	float adj_y_int = (-(bot.x - bot.tar_x) / (bot.y - bot.tar_y) * tcpy + tcpx
			+ (model->cur_pose.x - model->goTo.x) / (model->cur_pose.y - model->goTo.y) * model->cur_pose.y
			- model->cur_pose.x)
			/ ((model->cur_pose.x - model->goTo.x) / (model->cur_pose.y - model->goTo.y)
					- (bot.x - bot.tar_x) / (bot.y - bot.tar_y));
	if ((bot.y - bot.tar_y) / (bot.x - bot.tar_x) > 0)
	{
		return -calcDis(adj_x_int, adj_y_int, _x_int, _y_int);
	} else
	{
		return calcDis(adj_x_int, adj_y_int, _x_int, _y_int);
	}
}
/*
 void Rules::checkForProblems()
 {
 if (model.priority->size() != (int) UNUSED)
 {
 std::cout << "--------------ERROR: priority vector from <hawk_sim> does not have "
 "the priorities for each state in <enum State>------------------" << std::endl;
 }
 }*/
