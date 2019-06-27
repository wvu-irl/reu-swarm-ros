#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

Rules::Rules(Model _model)
{
	model = _model;
}

Rules::stateLoop()
{
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

	}
	return final_vel;
}
