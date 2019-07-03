#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include <map>
#include <string>
#include <math.h>
#include <iostream>

Rules::Rules()
{
	state = "explore";
}

Rules::Rules(Model _model) : model(_model)
{
	state = "explore";
}

//===================================================================================================================\\

AliceStructs::vel Rules::stateLoop()
{
	enum StringValue
	{
		explore, blocked, charge, find_food, find_updraft
	};
	std::map<std::string, StringValue> s_mapStringValues;
	switch (s_mapStringValues[state])
	{
	case explore:
		Explore();
		break;
	case blocked:
		avoidCollisions();
		break; /*
	case charge:
		Charge();
		break;
	case find_food:
		findFood();
		break;
	case find_updraft:
		findUpdraft();
		break; */
	}
	return final_vel;
}

//===================================================================================================================\\


void Rules::avoidCollisions()
{
	float tf = atan2(model.goTo.y/model.goTo.x);
	std::vector<std::pair<float, float>> dead_zones = findDeadZones();
	findAngle(tf, dead_zones);
	sort(testers.begin, testers.end);
	final_vel.dir = testers.front;
	final_vel.mag = 1;
	if (final_vel.dir == atan2(model.goTo.y/model.goTo.x))
	{
		state = "explore";
	}
}

void Rules::Explore()
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
	checkBlocked;
	final_vel.dir = atan2(model.goTo.y, model.goTo.x);
	final_vel.mag = 1;
}

void Rules::Charge()
{
	//To implement
}

void Rules::findFood()
{
	//To implement
}

void findUpdraft()
{
	//To implement
}

//===================================================================================================================\\

float Rules::calcDis(float _x1, float _y1, float _x2, float _y2)
{
	return pow(pow(_x1 - _x2, 2) + pow(_y1 - _y2, 2), 0.5);
}

bool checkBlocked()
{
	std::vector<std::pair<float, float>> dead_zones = findDeadZones;
	for (auto& zone : findDeadZones)
	{
		if (zone.first < atan2(model.goTo.y/model.oTo.x) < zone.second ||
				zone.first > atan2(model.goTo.y/model.oTo.x) > zone.second)
		{
			state = "blocked";
			return true;
		}
	}
	return false;
}

void Rules::findAngle(float tf, std::vector<std::pair<float, float>> dead_zones)
{
	std::vector<float> right;
	std::vector<float> left;
	for (auto& zone : dead_zones)
	{
		if (zone.first < tf < zone.second)
		{
			right.push_back(zone.first);
			left.push_back(zone.second);
		} else if (zone.second < tf < zone.first)
		{
			right.push_back(zone.second);
			left.push_back(zone.first);
		}
	}
	if (right.size() + left.size() == 0)
	{
		testers.push_back(tf);
	}
	else
	{
		right.sort();
		left.sort();
		find_angle(right.begin, dead_zones);
		find_angle(left.begom, dead_zones);
	}
}

std::vector<std::pair<float, float>> Rules::findDeadZones()
{
	float tf = atan2(model.goTo.y/model.goTo.x);
	std::vector<std::pair<float, float>> dead_zones;
	for (auto& obs : model.obstacles)
	{
		float temp_a_x = model.cur_pose.x - obs.x_off;
		float temp_a_y = model.cur_pose.y - obs.y_off;
		std::pair<float, float> aoe;
		float plus = (2*temp_a_x*temp_a_y + //This is the quadratic formula. It's just awful
				pow(pow(-2*temp_a_x*temp_a_y, 2) - 4*(pow(temp_a_x, 2) - pow(obs.x_rad, 2))*(pow(temp_a_y, 2) + pow(obs.y_rad, 2))), 0.5)
						/ (2*(pow(temp_a_x, 2) + pow(obs.x_rad, 2)));
		float neg = (2*temp_a_x*temp_a_y - //This is the quadratic formula again, this time the minus half
				pow(pow(-2*temp_a_x*temp_a_y, 2) - 4*(pow(temp_a_x, 2) - pow(obs.x_rad, 2))*(pow(temp_a_y, 2) + pow(obs.y_rad, 2))), 0.5)
						/ (2*(pow(temp_a_x, 2) + pow(obs.x_rad, 2)));
		aoe.first = atan(plus) - atan2(temp_a_y, temp_a_x) + atan2(obs.x_off, obs.y_off);
		aoe.second = atan(neg) - atan2(temp_a_y, temp_a_x) + atan2(obs.x_off, obs.y_off);
		dead_zones.push_back(aoe);
	}
	return dead_zones;
}
