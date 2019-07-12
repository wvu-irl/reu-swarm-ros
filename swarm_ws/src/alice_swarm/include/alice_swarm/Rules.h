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
	 * TODO
	 * Implement neighbor collisions. See picture on phone for math. Should be doable.
	 * Simulate
	 * Implement other rules (should also be doable)
	 * Document code
	 */

public:

	Rules();

	Rules(Model _model);

	enum State{REST, CHARGE, CONTOUR, TARGET, EXPLORE};
	State state;
	std::string collision_state;
	AliceStructs::vel final_vel;
	std::vector<float> testers;
	Model model;
	float margin = 2*model.SIZE + model.SAFE_DIS;

	/*
	 * Helper method to find the distance between two points
	 */
	float calcDis(float _x1, float _y1, float _x2, float _y2);

	/*
	 * Helper method to calculate the roots of a quadratic equation
	 */
	std::pair<float, float> calcQuad(float a, float b, float c);

	/*
	 * Checks whether there's an obstacle in the robot's path
	 */
	bool checkBlocked();

	//std::string checkBattery(std::string state);//checks to make sure battery has sufficient charge.
	/*
	 * Finds an adjusted angle to drive at given a set of blocked zones
	 */
	void findAngle(float tf, std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones);

	/*
	 * Finds the set of angles at which the robot will collide with an obstacle
	 */
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> findDeadZones();

	/*
	 * Avoids other robots dynamically
	 */
	bool avoidNeighbors();

//===================================================================================================================\\

	/*
	 * Makes Alice explore new territory
	 */
	void explore();

	/*
	 * Makes Alice stop moving
	 */
	void rest();

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
	 * Makes Alice seek higher elevations
	 */
	void findContour();

	/*
	 * The part of obstacle avoidance that always runs. Checks if any part of the environment has changed enough (or the robot's divergence from the predicted path)
	 * require a recalculation of the path.
	 */
	bool checkCollisions();

	/*
	 * Used to update the distance and direction of the command vector, without changing the true way-point.
	 */
	void updateVel();

	/*
	 * Changes the way-point's true location. Most simplistically, does this if the way point has been reached.
	 * Can also change the waypoint if certain priority checks are satisfied.
	 * Returns true is the waypoint was changed.
	 */
	bool updateWaypoint();

	/*
	 * Determines which state Alice is in
	 */
	AliceStructs::vel stateLoop(Model &_model);

	/*
	 * Checks whether two robots will collide
	 */
	float checkTiming(float _x_int, float _y_int, AliceStructs::neighbor bot);

};

#endif /* RULES_H */
