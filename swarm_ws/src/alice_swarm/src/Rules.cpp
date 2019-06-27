#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

Rules::Rules(Model _model)
{
	model = _model;
}

AliceStructs::vel Rules::stateLoop()
{
	priorState = state;
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
	string collision_state = model.getCollisionState();
	switch (collision_state)
	{
	case "Free":
		state = priorState;
		to_return.mag = -1;
		return to_return;
	case "Blocked":
		if ()
		break;
	case "Recontre":

	case "Rendevous":

	}
}
