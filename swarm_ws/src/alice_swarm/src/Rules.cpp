#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

Rules::Rules(Model _model)
{
	model = _model;
	state = "Explore";
}

AliceStructs::vel Rules::stateLoop()
{
	if (isBlocked())
	{
		state = "blocked";
	}
	switch (state)
	{
	case "Explore":
		Explore();
		break;
	case "blocked":
		avoidCollisions();
		break;
	case "Charge":
		Charge();
		break;
	case "findFood":
		findFood();
		break;
	case "findUpdraft":
		findUpdraft();
		break;
	}
	return final_vel;
}

float Rules::calcDis(_x1, _y1, _x2, _y2)
{
	return pow(pow(_x1 - x_2, 2) + pow(_y1 - _y2, 2), 0.5);
}

void Rules::avoidCollsions()
{
	switch (collision_state)
	{
	case "Blocked":
		final_vel.mag = 0;
		final_vel.dir = 0;
		break;
	case "Recontre":
		//To implement
	case "Rendevous":
		//To implement
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
	final_vel.dir = atan2(goTo.y, goTo.x);
	final_vel.mag = 1;
	return final_vel;
}

void Rules::avoidCollisions()
{
	//To implement
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

}
