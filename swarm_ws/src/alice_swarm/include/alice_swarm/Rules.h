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

	/*
	 * Checks that the battery is not bellow acceptable levels.
	 */
	bool checkBattery(std::string state);

	bool changedPriorities(); //checks if the highest prior rule has changed.

	/*
	 * Finds an adjusted angle to drive at given a set of blocked zones
	 */
	void findAngle(float tf, std::vector<std::pair<float, float>> dead_zones);

	/*
	 * Finds the set of angles at which the robot will collide with an obstacle
	 */
	std::vector<std::pair<float, float>> findDeadZones();

	/*
	 * Helper method to find the distance between two points
	 */
	float calcDis(float _x1, float _y1, float _x2, float _y2);

	/*
	 * Used to update the distance and direction of the command vector, without changing the true way-point.
	 */
	void updateVel(AliceStructs::vel *_fv);

	/*
	 * Avoids other robots dynamically
	 */
	void avoidNeighbors();

	/*
	 * Makes Alice go to a target
	 */
	void goToTar();

	/*
	 * Prevents Alice from hitting obstacles or other robots
	 */
	void avoidCollisions();

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

	//============================================Priority Rules================================================================
	/*
	 * Makes Alice explore new territory
	 */
	void explore();

	/*
	 * Makes Alice stop moving
	 */
	void rest();

	/*
	 * Makes Alice find a charging station
	 */
	void Charge();

	/*
	 * Makes Alice seek higher elevations
	 */
	void findContour();
	//=========================================================================================================================

};

#endif /* RULES_H */
