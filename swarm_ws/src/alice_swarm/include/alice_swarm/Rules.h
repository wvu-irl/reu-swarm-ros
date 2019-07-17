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

	enum State{REST, CHARGE, CONTOUR, TARGET, EXPLORE, UNUSED}; // always keep UNUSED at the back of the array.
	State state;
	std::string collision_state;
	AliceStructs::vel final_vel;
	std::vector<float> testers;
	Model model;
	float margin = 2*model.SIZE + model.SAFE_DIS;

	/*
	 * Determines which state Alice is in
	 */
	Model stateLoop(Model &_model);
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

	/*
	* Checks whether two robots will collide
	*/
	float checkTiming(float _x_int, float _y_int, AliceStructs::neighbor bot);

	/*
	 * The part of obstacle avoidance that always runs. Checks if any part of the environment has changed enough (or the robot's divergence from the predicted path)
	 * require a recalculation of the path.
	 */
	bool checkCollisions();

	bool shouldLoop();

	/*
	 * Checks that the battery is not bellow acceptable levels.
	 */
	bool checkBattery(std::string state);

	bool charged();
	/*
	 * function to check for potential pit falls. Add cases for hard to find bugs.
	 */
	void checkForProblems();

	/*
	 * Finds an adjusted angle to drive at given a set of blocked zones
	 */
	void findPath(float tf, std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones);

	/*
	 * Finds the set of angles at which the robot will collide with an obstacle
	 */
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> findDeadZones();

	/*
	 * Used to update the distance and direction of the command vector, without changing the true way-point.
	 */
	void updateVel(AliceStructs::vel *_fv);

	/*
	 * Avoids other robots dynamically
	 */
	bool avoidNeighbors();

	/*
	 * Prevents Alice from hitting obstacles or other robots
	 */
	void avoidCollisions();


	AliceStructs::pnt charge2();
	//============================================Priority Rules================================================================
	/*
	 * Makes Alice stop moving
	 */
	AliceStructs::pnt rest();

	/*
	 * Makes Alice find a charging station
	 */
	AliceStructs::pnt charge();

	/*
	 * Makes Alice seek higher elevations
	 */
	void findContour();

	/*
	 * Makes Alice go to a target
	 */
	AliceStructs::pnt goToTar();

	/*
	 * Makes Alice explore new territory
	 */
	void explore();
	//=========================================================================================================================

};

#endif /* RULES_H */
