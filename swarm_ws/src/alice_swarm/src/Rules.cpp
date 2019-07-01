#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include <map>
#include <string>
#include <math.h>
#include <iostream>

Rules::Rules(Model _model) : model(_model)
{
	state = "Explore";
}

AliceStructs::vel Rules::stateLoop()
{
	enum StringValue
	{
		explore, blocked, charge, find_food, find_updraft
	};
	std::map<std::string, StringValue> s_mapStringValues;
	if (isBlocked())
	{
		state = "blocked";
	}
	switch (s_mapStringValues[state])
	{
	case explore:
		Explore();
		break;
	case blocked:
		avoidCollisions();
		break;
	case charge:
		Charge();
		break;
	case find_food:
		findFood();
		break;
	case find_updraft:
		findUpdraft();
		break;
	}
	return final_vel;
}

float Rules::calcDis(float _x1, float _y1, float _x2, float _y2)
{
	return pow(pow(_x1 - _x2, 2) + pow(_y1 - _y2, 2), 0.5);
}

void Rules::avoidCollisions()
{
	enum StringValue
	{
		blocked, recontre, rendevous
	};
	std::map<std::string, StringValue> s_mapStringValues;
	switch (s_mapStringValues[collision_state])
	{
	case blocked:
		final_vel.mag = 0;
		final_vel.dir = 0;
		break;
	case recontre:
		//To implement
		break;
	case rendevous:
		//To implement
		break;
	}
}

void Rules::Explore()
{
	float temp = 0;
	for (auto& tar : model.targets)
	{
		float check = calcDis(tar.x, tar.y, model.cur_pose.x, model.cur_pose.y);
		if (check < temp)
		{
			temp = check;
			model.goTo = tar;
		}
	}
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
