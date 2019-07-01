/*
 * Rules.h
 *
 * A finite state machine that takes data from the model and outputs vectors
 *
 * Author: Casey Edmonds-Estes
 */

#ifndef RULES_H
#define RULES_H

#include "aliceStructs.h"
#include "Model.h"


class Rules
{


private:

	std::string state;
	std::string collision_state;
	AliceStructs::vel final_vel;
	Model model;

	/*
	 * Calculates distances
	 */
	float calcDis(float _x1, float _y1, float _x2, float _y2);

	/*
	 * Checks if Alice needs to go into collision avoidance mode
	 */
	bool isBlocked();

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

public:


	Rules();
	Rules(Model _model);

	/*
	 * Determines which state Alice is in
	 */
	AliceStructs::vel stateLoop();

};

#endif /* RULES_H */
