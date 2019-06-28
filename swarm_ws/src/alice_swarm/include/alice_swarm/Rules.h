/*
 * Rules.h
 *
 * A finite state machine that takes data from the model and outputs vectors
 *
 * Author: Casey Edmonds-Estes
 */

#ifndef RULES_H
#define RULES_H

#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include "alice_swarm/Alice.h"


class Rules
{
public:

	Rules(Model _model);

	/*
	 * Determines which state Alice is in
	 */
	AliceStructs::vel stateLoop();


private:

	Model model;
	String state;
	string collision_state
	AliceStructs::vel final_vel;

	/*
	 * Makes Alice explore new territory
	 */
	void Explore();

	/*
	 * Prevents Alice from hitting obstacles or other robots
	 */
	void avoidCollisions();

	/*
	 * Makes Alice find a charging station
	 */
	void Charge();

	/*
	 * Makes Alice find food
	 */
	void findFood();

	/*
	 * Makes Alice seek higher elevations
	 */
	void findUpdraft();
};

#endif /* RULES_H */
