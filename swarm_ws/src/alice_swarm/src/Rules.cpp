#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include <map>
#include <string>
#include <math.h>
#include <iostream>
#include <bits/stdc++.h>

#define DEBUG_NEI_AVD 1
#if DEBUG_NEI_AVD
#include <chrono>
using namespace std::chrono;

#include <string.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define PRINTF_DBG(form, dat...) \
	printf(("[%d.%09d] (%s:%d) " + std::string(form) + "\n").c_str(),\
			duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count(),\
			duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count() % 1000000000\
			, __FILENAME__, __LINE__, dat\
			)

#define PRINT_DBG(form) \
	printf(("[%d.%09d] (%s:%d) " + std::string(form) + "\n").c_str(),\
			duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count(),\
			duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count() % 1000000000\
			, __FILENAME__, __LINE__\
			)
#endif

#define DEBUG_schange 0
#define DEBUG_charging 0

typedef AliceStructs::vector_2f vector2f_t;

#define CURR_VECT ((vector2f_t) { model->cur_pose.x, model->cur_pose.y, model->goTo.x - model->cur_pose.x,\
				model->goTo.y - model->cur_pose.y, true})

#define FAIL_VECT ((vector2f_t){0,0,0,0,false})

#define magnitude(a) ((double) sqrt(pow(a.dx - a.x,2) + pow(a.dy - a.y, 2)))
#define getUVect(a) (magnitude(a) != 0 ? (vector2f_t){a.x,a.y,a.dx / magnitude(a), a.dy / magnitude(a), true} : (vector2f_t){0,0,0,0, false})

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
	std::pair<float, float> cur_temp = _model.transformFir(_model.goTo.x,
			_model.goTo.y); //Shifts the frame of goTo to the current frame for calculations
	cur_go_to.x = cur_temp.first;
	cur_go_to.y = cur_temp.second;
	bool run = shouldLoop();
	if (run)
	{
		std::vector<AliceStructs::pnt> go_to_list;

		////////////////////////////////////////////////////////////////////////////////////////
		//                           RULES, THIS IS WHERE TEHY GO                             //
		////////////////////////////////////////////////////////////////////////////////////////

		// add other rules here
		go_to_list.push_back(findContour());
		go_to_list.push_back(goToTar());
		//go_to_list.push_back(charge());
		go_to_list.push_back(rest());
		go_to_list.push_back(explore());

		////////////////////////////////////////////////////////////////////////////////////////

		float temp = -1;
		int rulenum = -1;
		int rulenum2 = 0;
		for (auto &rule : go_to_list)
		{
			if (rule.z > temp)
			{
				temp = rule.z;

				cur_go_to.x = rule.x;
				cur_go_to.y = rule.y;
				rulenum = rulenum2;
//				std::cout << "theta: "<<atan2(model->goTo.y, model->goTo.x) << std::endl;
//				std::cout<<"(x,y): "<<model->goTo.x<<","<<model->goTo.y<<std::endl;
			}
			//}
			//	avoidCollisions();
//			float mag = sqrt(pow(model->goTo.x,2) + pow(model->goTo.y,2));
//			std::cout<<"(x,y): "<<model->goTo.x<<","<<model->goTo.y<<"| "<<mag<<std::endl;
//			std::cout<<"priority "<<temp<<std::endl;
//			std::cout<<"-----------------------------------"<<std::endl;
			rulenum2++;
		}
//		std::cout << "using rule " << rulenum << std::endl;
		_model.goTo.time = ros::Time::now(); //time stamps when the goto is created
	}
	goToTimeout();
	avoidNeighbors();
	std::pair<float, float> back_to_fir = _model.transformCur(cur_go_to.x,
			cur_go_to.y);
	_model.goTo.x = back_to_fir.first;
	_model.goTo.y = back_to_fir.second;
}

//===================================================================================================================
void Rules::goToTimeout()
{
	if (ros::Time::now().sec - model->goTo.time.sec > 8) //if it's been more than 8 seconds, move goTo
	{

		float mag = calcDis(cur_go_to.x, cur_go_to.y, 0, 0);

		//for handling bad goTo points (too far)
		cur_go_to.x *= 10 / mag;
		cur_go_to.y *= 10 / mag;

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
//------------------Basic Rules------------------------------------------
AliceStructs::pnt Rules::explore()
{
	AliceStructs::pose best;
	float ran = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //random float from 0 to 1
	std::pair<float, float> sum(-1, 2 * ran - 1); //adds tendency for robot to go forward with a slight tilt, just found useful.
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
	if (mag > 0.001) // set the magnitude to 10 unless the vector is 0 (or very close).
	{
		sum.first *= 10 / mag;
		sum.second *= 10 / mag;
	}
	AliceStructs::pnt to_return;
	to_return.x = -sum.first;
	to_return.y = -sum.second;
	to_return.z = 0.99 / pow(1 + model->archived_contour.size(), 0.5); //robot desires exploration when its' map is empty.
	return to_return;

}
AliceStructs::pnt Rules::rest() //gives a command to not move.
{
	//returns its' own location with a set priority
	AliceStructs::pnt go_to;
	go_to.x = 0;
	go_to.y = 0;
	if (model->energy <= 0)
		go_to.z = 1;
	else
		go_to.z = 0;
	return go_to;
}
AliceStructs::pnt Rules::goToTar()
{
	float temp = 10000;
	AliceStructs::pnt to_return;
	to_return.x = 0;
	to_return.y = 0;
	to_return.z = 0; //if no target is in range.
	if (model->energy > 0.8)
		return to_return; //if not hungry simply return
	for (auto &tar : model->targets) //check immediate vision
	{
		float check = calcDis(tar.x, tar.y, 0, 0);
		if (check < temp)
		{
			temp = check;
			to_return = tar;
			to_return.z = 0.999; //if there's food, the robot goes there unless it's dead
		}
	}
	for (auto &tar : model->archived_targets) //check archives
	{
		std::pair<float, float> temppos = model->transformFir(tar.x, tar.y);
		float check = calcDis(temppos.first, temppos.second, 0, 0);
		if (check < temp)
		{
			temp = check;

			to_return.x = temppos.first;
			to_return.y = temppos.second;
			to_return.z = 0.999; //if there's food, the robot goes there unless it's dead
		}
	}
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
	int crowding = 1;	//intended on telling if the contour area is already crowded

	if (init)
	{
		std::pair<float, float> go = model->transformFir(best.x, best.y);
		to_return.x = go.first;
		to_return.y = go.second;

		to_return.z = 1 - model->energy;	//sets the initial priority to its' hunger

		to_return.z /= crowding; //divide by the number of robots crowding the area
		to_return.z *= (log10(best.z / model->cur_pose.z)); //scale by the appreciable change in the contour.
		if (to_return.z >= 0.9)
			to_return.z = 0.9;
	}

	return to_return;
}
//-------------------Charging related rules----------------------------------
AliceStructs::pnt Rules::charge()
{
	AliceStructs::pnt go_to;
	float dx;
	float dy;
	float min_sep = model->min_sep;
	float check_sep; //sep distance
	bool odd_man_out = false;

#if DEBUG_charging
	for(int j = 0; j < model->abs_chargers->size(); j ++)
	{
		std::cout<<"in rules "<<j<<std::endl;
		std::cout<<(model->abs_chargers->at(j).occupied? "true" : "false")<<std::endl;
	}
#endif

	if ((!availableChargers() && !model->committed)
			|| (model->battery_state == CHARGED && model->battery_lvl == 0)) //makes this a non op when no chargers available.
	{
		odd_man_out = true;
	}

	if (model->abs_chargers->size() > 0 && !odd_man_out)
	{
//		std::cout<<"OP for bot: "<<model->name<<std::endl;
//		std::cout<<"availableChargers(): "<<(availableChargers() ? "true" : "false")<<std::endl;
		if (!model->charge2) //runs code for first stage if 2nd not initiated.
		{
			for (int i = 0; i < model->rel_chargers.size(); i++) //checks each charger
			{
				if ((!model->abs_chargers->at(i).occupied) && (!model->committed)) //charger is open or charger not committed
				{
					dx = model->rel_chargers.at(i).target_x;
					dy = model->rel_chargers.at(i).target_y;
					check_sep = sqrt(pow(dx, 2) + pow(dy, 2)); //check separation distance
//					std::cout<<"charger: "<<i<<":"<<"dx,dy: "<<dx<<","<<dy<<"|"<<check_sep<<std::endl;
					if (check_sep < min_sep) //if the closest
					{
						model->closest_pos = i; //saves pos of closest
						min_sep = check_sep; //updates min_sep
						model->min_sep = min_sep;
					}
				}
//				std::cout<<"why you break? "<<i<<std::endl;
			}
//			std::cout<<model->closest_pos<<" :closest pos"<<std::endl;
			go_to.x = model->rel_chargers.at(model->closest_pos).target_x; //sets pos of closest charger as target.
			go_to.y = model->rel_chargers.at(model->closest_pos).target_y;

			if ((pow(go_to.x, 2) + pow(go_to.y, 2)) < pow(model->SIZE / 5, 2)
					&& (model->committed)) //initiates charge2 after it reaches first waypoint.
			{
				model->charge2 = true;
			}
//			std::cout<<"the battery level is: "<<model->battery_lvl<<std::endl;
			if ((model->battery_lvl < 3.8)) //assign priority
			{
				go_to.z = 2; //given highest priority
				model->abs_chargers->at(model->closest_pos).occupied = true; //charger is "checked out".
				model->committed = true;
				model->rel_chargers.at(model->closest_pos).occupied = true;
			}
			else
			{
//				std::cout<<"saftey is running"<<std::endl;
//				//makes sure that the reset sticks. There is lag in reseting these vars sometimes.
				model->abs_chargers->at(model->closest_pos).occupied = false;
				model->rel_chargers.at(model->closest_pos).occupied = false;
				model->committed = false;
				model->charge2 = false;
				go_to.z = 0;
			}
		}
		else if (model->charge2) //runs code for second waypoint.
		{
#if DEBUG_charging
			std::cout<<"++++++++++++++++++++++++++++++\n";
			std::cout<<"charge2 activated"<<std::endl;
			std::cout<<"the battery level is: "<<model->battery_lvl<<std::endl;
//			std::cout<<(model->abs_chargers->at(model->closest_pos).occupied ? "true" : "false")<<std::endl;
			std::cout<<"charged()"<<(charged() ? "true" : "false")<<std::endl;
#endif
			go_to = charge2();

			if (charged()) //resets charging vars, and makes priority of the generated point zero.
			{
#if DEBUG_charging
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
	}
	else
	{
//		std::cout<<"No Op for bot"<<std::endl;
//		std::cout<<"No Op for bot: "<<model->name<<std::endl;
		model->min_sep = 1000.0;
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
	}
	else
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
	while (i < model->abs_chargers->size())
	{
		if (!model->abs_chargers->at(i).occupied)
		{
			result = true;
			i = model->abs_chargers->size() + 1;
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
//-----------------------------------------------------------------------------------------

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
		if (pow(pow(zone.second.x_off, 2) + pow(zone.second.y_off, 2), 0.5)
				< pow(pow(waypnt.x, 2) + pow(waypnt.y, 2), 0.5))
		{
			//std::cout << "it's on the proper side of me" << std::endl;
			//std::cout << fabs(tf) << " first " << fabs(zone.first.first) << " second " << fabs(zone.first.second) << std::endl;
			if (((zone.first.first < tf && tf < zone.first.second)
					|| (zone.first.second < tf && tf < zone.first.first))
					|| ((fabs(zone.first.first - zone.first.second) > M_PI / 2)
							&& ((fabs(tf) + M_PI / 12 > fabs(zone.first.first))
									&& (fabs(tf) + M_PI / 12 > fabs(zone.first.second)))))
			{
				std::cout << "it's in my way" << std::endl;
				blockers.push_back(zone);
			}
		}
	}
	std::pair<std::pair<float, AliceStructs::obj>,
			std::pair<float, AliceStructs::obj>> angles;
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
	std::cout << "first " << angles.first.first << " second "
			<< angles.second.first << " tf " << tf << std::endl;
	if (fabs(fabs(angles.first.first) - fabs(tf))
			< fabs(fabs(angles.second.first) - fabs(tf)))
	{
		final = angles.first;
	}
	else
	{
		final = angles.second;
	}
	std::cout << final.first << std::endl;
	std::cout << "x: " << final.second.x_off << " y: " << final.second.y_off
			<< std::endl;
	std::cout << "q: " << final.second.x_rad << " p: " << final.second.y_rad
			<< std::endl;
	if (final.first != tf)
	{
		float m = tan(final.first);
		std::cout << "m: " << m << std::endl;
		to_return.x = 0.5
				* (2 * final.second.x_off / pow(final.second.x_rad, 2)
						+ 2 * final.second.y_off * m / pow(final.second.y_rad, 2))
				/ (1 / pow(final.second.x_rad, 2)
						+ pow(m, 2) / pow(final.second.y_rad, 2));
		to_return.y = 0.5
				* (2 * final.second.y_off / pow(final.second.y_rad, 2)
						+ 2 * final.second.x_off / (m * pow(final.second.x_rad, 2)))
				/ (1 / pow(final.second.y_rad, 2)
						+ 1 / (pow(m, 2) * pow(final.second.x_rad, 2)));
		std::cout << "test: " << (2 * final.second.y_off * m) << std::endl;
		std::cout << "x: " << to_return.x << " y: " << to_return.y << std::endl;
		std::cout << atan2(to_return.y, to_return.x) << std::endl;
	}
	else
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
		std::pair<float, float> aoe = calcQuad(
				pow(obs.x_off, 2) - pow(obs.x_rad + model->SIZE + model->SAFE_DIS, 2),
				-2 * obs.x_off * obs.y_off,
				pow(obs.y_off, 2) - pow(obs.y_rad + model->SIZE + model->SAFE_DIS, 2));
		//Unfortunately, trig can't handle quadrents very well, so we have to do it ourselves
		if (obs.y_off - obs.y_rad - model->SIZE - model->SAFE_DIS < 0)
		{
			if (obs.x_off + obs.x_rad + model->SIZE + model->SAFE_DIS < 0)
			{
				//std::cout << "top right" << std::endl;
				aoe.first = -M_PI + atan(aoe.first);
			}
			else
			{
				//std::cout << "top left" << std::endl;
				aoe.first = atan(aoe.first);
			}
		}
		else
		{
			if (obs.x_off - obs.x_rad - model->SIZE - model->SAFE_DIS < 0)
			{
				//std::cout << "bottom right" << std::endl;
				aoe.first = M_PI + atan(aoe.first);
			}
			else
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
			}
			else
			{
				//std::cout << "top left" << std::endl;
				aoe.second = atan(aoe.second);
			}
		}
		else
		{
			if (obs.x_off + obs.x_rad + model->SIZE + model->SAFE_DIS < 0)
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
		std::cout << "dead zones: " << aoe.first << " - " << aoe.second
				<< std::endl;
		std::pair<std::pair<float, float>, AliceStructs::obj> to_push;
		to_push.first = aoe;
		to_push.second = obs;
		dead_zones.push_back(to_push);
	}
	return dead_zones;
}

vector2f_t findIntersect(vector2f_t a, vector2f_t b)
{
	double x_intercept;
	vector2f_t vect;
	if (a.dx == 0 && b.dx == 0)
	{
		return FAIL_VECT;
	}
	else if (a.dx == 0)
	{
		x_intercept = a.x;
		vect.y = (b.dy / b.dx) * (x_intercept - b.x) + b.y;
	}
	else if (b.dx == 0)
	{
		x_intercept = b.x;
		vect.y = (a.dy / a.dx) * (x_intercept - a.x) + a.y;
	}
	else
	{
		double m0 = a.dy / a.dx;
		double m1 = b.dy / b.dx;

		if (m0 == m1)
		{
			return FAIL_VECT;
		}

		x_intercept = (b.y - a.y + a.x * m0 - b.x * m1) / (m0 - m1);
		vect.y = (a.dy / a.dx) * (x_intercept - a.x) + a.y;
	}

	vect.x = x_intercept;
	vect.dx = 0;
	vect.dy = 0;
	vect.valid = true;
	return vect;
}

bool Rules::avoidNeighbors()
{
	bool to_return = false;
	for (size_t i = 0; i < model->neighbors.size(); i++)
	{
		AliceStructs::neighbor &bot = model->neighbors[i];
		struct AliceStructs::vector_2f vec_bot;
		vec_bot.x = bot.x;
		vec_bot.y = bot.y;
		vec_bot.dx = (double) bot.tar_x - (double) bot.x;
		vec_bot.dy = (double) bot.tar_y - (double) bot.y;
		vec_bot.valid = true;
		vector2f_t vec_crr = CURR_VECT;
		vec_bot = getUVect(vec_bot);
		vec_crr = getUVect(vec_crr);
		vector2f_t circle_center = findIntersect(vec_bot, vec_crr);

		if (!circle_center.valid)
		{
#if DEBUG_NEI_AVD
			PRINT_DBG("invalid");
#endif
			continue;
		}

		double self_tti = circle_center.x / vec_crr.dx;
		double bot_tti = circle_center.x / vec_bot.dx;

		if (abs(self_tti - bot_tti)
				< checkTiming(circle_center, vec_bot) / model->MAX_LV)
		{
			// TODO reverse the order of these to make the parallel
			// parallel to the curr not the bot
			vector2f_t conjunct = { circle_center.x, circle_center.y, -vec_bot.dy,
					vec_bot.dx, true }; // same point in perpendicular direction
			conjunct = getUVect(conjunct);
			vector2f_t new_center = { conjunct.dx * -margin + conjunct.x, conjunct.dy
					* -margin + conjunct.y, 0, 0, true };

			///
			if (calcDis(new_center.x, new_center.y, model->cur_pose.x,
					model->cur_pose.y)
					< calcDis(model->goTo.x, model->goTo.y, model->cur_pose.x,
							model->cur_pose.y))
			{
				vector2f_t parallel = { new_center.x, new_center.y, center.dx,
						center.dy, true };

				vector2f_t adj = findIntersect(vec_bot, parallel);
				model->goTo.x = adj.x;
				model->goTo.y = adj.y;
#if DEBUG_NEI_AVD
				PRINTF_DBG("Got nieghbor %d -- %d", i, model->neighbors.size() - 1);
				PRINTF_DBG("New waypoint: (%lf, %lf)", new_center.x, new_center.y);
#endif
				to_return = true;
			}
		}
	}
	return to_return;
}

float Rules::checkTiming(vector2f_t center, vector2f_t bot)
{
	vector2f_t conjunct = { center.x, center.y, -bot.dy, bot.dx }; // same point in perpendicular direction
	conjunct = getUVect(conjunct);
	vector2f_t new_center = { conjunct.dx * -margin + conjunct.x, conjunct.dy
			* -margin + conjunct.y, 0, 0, true };

	vector2f_t parallel =
			{ new_center.x, new_center.y, center.dx, center.dy, true };
	vector2f_t vec_crr = CURR_VECT;
	vec_crr = getUVect(vec_crr);

	vector2f_t adj = findIntersect(vec_crr, parallel);

	if (bot.dy > 0)
	{
		return -calcDis(adj.x, adj.y, center.x, center.y);
	}
	else
	{
		return calcDis(adj.x, adj.y, center.x, center.y);
	}
}
