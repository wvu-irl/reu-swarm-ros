#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include <map>
#include <string>
#include <math.h>
#include <iostream>
#include <bits/stdc++.h>

Rules::Rules()
{
	state = "explore";
}

Rules::Rules(Model _model) :
		model(_model)
{
	state = "explore";
}

//===================================================================================================================\\

AliceStructs::vel Rules::stateLoop(Model &_model)
{

	model = _model;
	updateWaypoint();
	if (checkCollisions()) avoidCollision();
	updateVel();
//
//	state = checkBattery(state);
//	checkBlocked();
//	if (state == "goToTar")
//	{
//		goToTar();
//	}
//	else if (state == "blocked")
//	{
//		avoidCollisions();
//	}
//
//	else if (state == "needs_charging")
//	{
//		Charge();
//	}
//	else if(state == "charging")

//	checkBlocked();
//	if (state == "goToTar")
//	{
//		goToTar();
//	}
//	else if (state == "blocked")
//	{
//		avoidCollisions();
//	}/*
//	case charge:
//		Charge();
//		break;
//	case find_food:
//		findFood();
//		break;
//	case find_updraft:
//		findUpdraft();
//		break; */

	findContour();
	//explore();

	return final_vel;
}

//===================================================================================================================
void Rules::avoidCollisions()
{
	float tf = atan2(model.goTo.y, model.goTo.x);
	std::vector<std::pair<float, float>> dead_zones = findDeadZones();
	findAngle(tf, dead_zones);
	final_vel.mag = 1;
	if (final_vel.dir == atan2(model.goTo.y, model.goTo.x))
	{
		state = "explore";
	}
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

void Rules::Charge()
{
	float min_sep = 1000.0;
	float check_sep;
	int closest_pos;
	for (int i=0; i < model.chargers.size(); i++)
		{
			if(!model.chargers.at(i).occupied) //charger is open
			{
				check_sep = sqrt(pow(0,2) + pow(0,2)); //check seperation distance
				if(check_sep < min_sep)
				{
					closest_pos = i; //saves pos of closest
					min_sep = check_sep; //updates min_sep
				}
			}
		}
		model.chargers.at(closest_pos).occupied = true;

//	for (int i=0; i < model.chargers->charger.size(); i++)
//	{
//		if(!model.chargers->charger.at(i).occupied) //charger is open
//		{
//			check_sep = sqrt(pow(0,2) + pow(0,2)); //check seperation distance
//			if(check_sep < min_sep)
//			{
//				closest_pos = i; //saves pos of closest
//				min_sep = check_sep; //updates min_sep
//			}
//		}
//	}
//	model.chargers->charger.at(closest_pos).occupied = true;

	//make the bot go to some way point, overiding other directives.
	//way point should be .y, .x + 5 if on the left wall.
}

void Rules::goToTar()
{
	float temp = 10000;
	for (auto& tar : model.targets)
	{
		float check = calcDis(tar.x, tar.y, model.cur_pose.x, model.cur_pose.y);
		if (check < temp)
		{
			temp = check;
			model.goTo = tar;
		}
	}
	checkBlocked();
	final_vel.dir = atan2(model.goTo.y, model.goTo.x);
	final_vel.mag = 1;
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

//===================================================================================================================\\

float Rules::calcDis(float _x1, float _y1, float _x2, float _y2)
{
	return pow(pow(_x1 - _x2, 2) + pow(_y1 - _y2, 2), 0.5);
}

bool Rules::checkBlocked()
{
	for (auto& zone : findDeadZones())
	{
		if (zone.first < atan2(model.goTo.y, model.goTo.x) < zone.second
				|| zone.first > atan2(model.goTo.y, model.goTo.x) > zone.second)
		{
			state = "blocked";
			return true;
		}
	}
	return false;
}

std::string Rules::checkBattery(std::string state)
{
	float acceptable_lvl = model.battery_lvl * 0.2;
	if(model.battery_lvl < acceptable_lvl && state != "charging")
	{
		state = "needs_charging";
	}
	return state;
}
void Rules::findAngle(float tf, std::vector<std::pair<float, float>> dead_zones)
{
	std::vector<float> right;
	std::vector<float> left;
	for (auto& zone : dead_zones)
	{
		if (pow(pow(model.cur_pose.x, 2) + pow(model.cur_pose.y, 2), 0.5) <= // Checks that the obstacle is on the correct side of the bot
				pow(pow(model.cur_pose.x - model.goTo.x, 2) + pow(model.cur_pose.y - model.goTo.x, 2), 0.5))
		{
			if (tf < 0) // Checks which side of the bot the obstacle is on
			{
				if (zone.first > tf)
				{
					right.push_back(zone.first);
				}
			} else
			{
				if (zone.second < tf)
				{
					left.push_back(zone.second);
				}

			}
		}
	}
	if (right.size() > 0 && left.size() > 0)
	{
		sort(right.begin(), right.end());
		sort(left.begin(), left.end());
		std::cout << left.front() << std::endl;
		if (right.front() > left.front())
		{
			final_vel.dir = left.front();
		} else
		{
			final_vel.dir = right.front();
		}
	} else if (right.size() > 0)
	{
		sort(right.begin(), right.end());
		final_vel.dir = right.front();
	} else if (left.size() > 0)
	{
		sort(left.begin(), left.end());
		std::cout << left.front() << std::endl;
		final_vel.dir = left.front();
	} else
	{
		final_vel.dir = tf;
	}
}

std::vector<std::pair<float, float>> Rules::findDeadZones()
{
	std::vector<std::pair<float, float>> dead_zones;
	for (auto& obs : model.obstacles)
	{
		float temp_a_x = model.cur_pose.x - obs.x_off;
		float temp_a_y = model.cur_pose.y - obs.y_off;
		std::pair<float, float> aoe;
		float plus = (2 * temp_a_x * temp_a_y + //This is the quadratic formula. It's just awful
				pow(
						pow(-2 * temp_a_x * temp_a_y, 2)
								- 4 * (pow(temp_a_y, 2) - pow(obs.y_rad, 2)) * (pow(temp_a_x, 2) - pow(obs.x_rad, 2)), 0.5))
				/ (2 * (pow(temp_a_x, 2) - pow(obs.y_rad, 2)));
		float neg = (2 * temp_a_x * temp_a_y - //This is the quadratic formula again, this time the minus half
				pow(
						pow(-2 * temp_a_x * temp_a_y, 2)
								- 4 * (pow(temp_a_y, 2) - pow(obs.y_rad, 2)) * (pow(temp_a_x, 2) - pow(obs.x_rad, 2)), 0.5))
				/ (2 * (pow(temp_a_x, 2) - pow(obs.y_rad, 2)));
		aoe.first = atan(plus);
		aoe.second = atan(neg);
		dead_zones.push_back(aoe);
	}
	return dead_zones;
}

void Rules::avoidNeighbors()
{
	for (auto& bot : model.neighbors)
	{

	}
}
