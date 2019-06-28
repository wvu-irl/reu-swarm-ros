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
	AliceStructs::vel final_vel;
	switch (state)
	{
	case "Explore":
		Explore();
		break;
	case "avoidCollisions":
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

void Rules::avoidCollsions()
{
	switch (collision_state)
	{
	case "Blocked":
		final_vel.mag = 0;
		final_vel.dir = 0;
		break;
	case "Recontre":

	case "Rendevous":

	}
}
