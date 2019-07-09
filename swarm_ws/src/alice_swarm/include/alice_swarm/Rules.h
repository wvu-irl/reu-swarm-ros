/*
 * Rules.h
 *
 * A finite state machine that takes data from the model and outputs vectors.
 *
 * Author: Casey Edmonds-Estes
 */

#ifndef RULES_H
#define RULES_H

#include "aliceStructs.h"
#include "Model.h"


class Rules
{

	/*
	 * To do:
	 * Implement neighbor collisions. See picture on phone for math. Should be doable.
	 * Simulate
	 * Implement other rules (should also be doable)
	 * Document code
	 */

public:

	Rules();

	Rules(Model _model);

	std::string state;
	std::string collision_state;
	AliceStructs::vel final_vel;
	std::vector<float> testers;
	Model model;

	/*
	 * Helper method to find the distance between two points
	 */
	float calcDis(float _x1, float _y1, float _x2, float _y2);


	/*
	 * Checks whether there's an obstacle in the robot's path
	 */
	bool checkBlocked();

	std::string checkBattery(std::string state);//checks to make sure battery has sufficient charge.
	/*
	 * Finds an adjusted angle to drive at given a set of blocked zones
	 */
	void findAngle(float tf, std::vector<std::pair<float, float>> dead_zones);

	/*
	 * Finds the set of angles at which the robot will collide with an obstacle
	 */
	std::vector<std::pair<float, float>> findDeadZones();

	/*
	 * Avoids other robots dynamically
	 */
	void avoidNeighbors(float tf);

//===================================================================================================================\\

	/*
	 * Makes Alice explore new territory
	 */
	void explore();

	/*
	 * Prevents Alice from hitting obstacles or other robots
	 */
	void avoidCollisions();

	/*
	 * Makes Alice find a charging station
	 */
	void Charge();

	/*
	 * Makes Alice go to a target
	 */
	void goToTar();

	/*
	 * Makes Alice seek high values on the contour map
	 */
	void findContour();

	void findContour();

	/*
	 * Determines which state Alice is in
	 */
	AliceStructs::vel stateLoop(Model &_model);

};

#endif /* RULES_H */
