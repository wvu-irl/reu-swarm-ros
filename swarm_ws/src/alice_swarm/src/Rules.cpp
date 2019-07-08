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

Rules::Rules(Model _model) : model(_model)
{
	state = "explore";
}

//===================================================================================================================\\

AliceStructs::vel Rules::stateLoop()
{
//	checkBattery();
	checkBlocked();
	if (state == "goToTar")
	{
		goToTar();
	}
	else if (state == "blocked")
	{
		avoidCollisions();
	}

//	else if (state == "needs_charging")
//	{
//
//	}
//	else if(state == "charging")
//	{
//
//	}

	/*
	case charge:
		Charge();
		break;
	case find_food:
		findFood();
		break;
	case find_updraft:
		findUpdraft();
		break; */
	return final_vel;
}

//===================================================================================================================\\


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

void Rules::Explore()
{
 // To implement
}

void Rules::Charge()
{
	//To implement
}

void Rules::findFood()
{
	/*
	AliceStructs::pnt best;
	float max_dis = 10; //to implement once model has the support
	for (auto& contour : model.archived_contour)
	{
		if (calcDis(model.cur_pose.x, model.cur_pose.y, contour.x, contour.y) < max_dis && contour.z > best.z)
		{
			best.x = contour.x;
			best.y = contour.y;
			best.z = contour.z;
		}
	}
	final_vel.dir = atan2(best.y - model.cur_pose.y, model.cur_pose.x - best.x);
	final_vel.mag = 1;
	model.goTo.x = best.x;
	model.goTo.y = best.y; */
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

void Rules::findUpdraft()
{
	AliceStructs::pnt best;
	float max_dis = 10; //to implement once model has the support
	for (auto& contour : model.archived_contour)
	{
		if (calcDis(model.cur_pose.x, model.cur_pose.y, contour.x, contour.y) < max_dis && contour.z > best.z)
		{
			best.x = contour.x;
			best.y = contour.y;
			best.z = contour.z;
		}
	}
	final_vel.dir = atan2(best.y - model.cur_pose.y, model.cur_pose.x - best.x);
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
		if (zone.first < atan2(model.goTo.y, model.goTo.x) < zone.second ||
				zone.first > atan2(model.goTo.y, model.goTo.x) > zone.second)
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
		if (pow(pow(model.cur_pose.x, 2) + pow(model.cur_pose.y, 2), 0.5) <= // Checks that the obstacle is on the correct side of the bot
				pow(pow(model.cur_pose.x - model.goTo.x, 2) + pow(model.cur_pose.y - model.goTo.x, 2), 0.5))
		{
			if (tf < 0) // Checks which side of the bot the obstacle is on
			{
				if (zone.first > tf)
				{
					right.push_back(zone.first);
				}
			}
			else
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
		}
		else
		{
			final_vel.dir = right.front();
		}
	}
	else if (right.size() > 0)
	{
		sort(right.begin(), right.end());
		final_vel.dir = right.front();
	}
	else if (left.size() > 0)
	{
		sort(left.begin(), left.end());
		std::cout << left.front() << std::endl;
		final_vel.dir = left.front();
	}
	else
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
		float	plus = (2*temp_a_x*temp_a_y + //This is the quadratic formula. It's just awful
				pow(pow(-2*temp_a_x*temp_a_y, 2) - 4*(pow(temp_a_y, 2) -
						pow(obs.y_rad, 2))*(pow(temp_a_x, 2) - pow(obs.x_rad, 2)), 0.5))
						/ (2*(pow(temp_a_x, 2) - pow(obs.y_rad, 2)));
		float neg = (2*temp_a_x*temp_a_y - //This is the quadratic formula again, this time the minus half
				pow(pow(-2*temp_a_x*temp_a_y, 2) - 4*(pow(temp_a_y, 2) -
						pow(obs.y_rad, 2))*(pow(temp_a_x, 2) - pow(obs.x_rad, 2)), 0.5))
						/ (2*(pow(temp_a_x, 2) - pow(obs.y_rad, 2)));
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
