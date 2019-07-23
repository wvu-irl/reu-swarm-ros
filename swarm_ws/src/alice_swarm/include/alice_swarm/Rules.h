/*
 * Rules.h
 *
 * A finite state machine that takes data from the model and outputs vectors.
 *
 * Author: Casey Edmonds-Estes
 * If you have questions, email me: cedmonds2018@gmail.com
 */

#ifndef RULES_H
#define RULES_H

#include "aliceStructs.h"
#include "Model.h"


struct vector_2f
{
	double x, y;
	double dx, dy;
};
typedef vector_2f vector2f_t;

#include <math.h>

#define magnitude(a) ((double) sqrt(pow(a.dx - a.x,2) + pow(a.dy - a.y, 2)))
#define getUVect(a) (magnitude(a) != 0 ? (vector2f_t){a.x,a.y,a.dx / magnitude(a), a.dy / magnitude(a)} : (vector2f_t){0,0,0,0})

class Rules
{
private:
	bool first_time=true; //We need this because sometimes somehow the goTo is initialized with a bullshit value
	void init(Model &_model);

	/*
	 * TODO
	 * Implement neighbor collisions. See picture on phone for math. Should be doable. <- reee
	 * Simulate
	 * Implement other rules (should also be doable)
	 * Document code
	 */

public:

	Rules();

	Rules(Model &_model);

	enum State{REST, CHARGE, CONTOUR, TARGET, EXPLORE, UNUSED}; // always keep UNUSED at the back of the array.
	State state;
	std::string collision_state;
	AliceStructs::vel final_vel;
	std::vector<float> testers;
	Model *model;
	float margin = 4;//2*model.SIZE + model.SAFE_DIS; this won't work the way you want it to, you have it defined before model is initialezed
	std::pair<float,float> cur_go_to;
	/*
	 * Determines which state Alice is in
	 */
	void stateLoop(Model &_model);
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
	float checkTiming(vector2f_t center, vector2f_t bot);

	/*
	 * The part of obstacle avoidance that always runs. Checks if any part of the environment has changed enough (or the robot's divergence from the predicted path)
	 * require a recalculation of the path.
	 */
	bool checkCollisions();

	/*
	 * Checks if the goTo point has not been updated for a set amount of time, and if so, reverses it.
	 * Meant for error handling bad goTo's and stuck robots.
	 */
	void goToTimeout();

	bool shouldLoop();

	/*
	 * Checks that the battery is not below acceptable levels.
	 */
	bool checkBattery(std::string state);

	AliceStructs::pnt charge2(); //carries out code for second waypoint.

	bool charged(); //checks if bot is charged. resets conditionals if true.

	bool availableChargers(); //checks if chargers are available.
	/*
	 * function to check for potential pit falls. Add cases for hard to find bugs.
	 */
	void checkForProblems();

	/*
	 * Finds an adjusted angle to drive at given a set of blocked zones
	 */
	AliceStructs::pnt findPath(AliceStructs::pnt waypnt, std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> dead_zones);

	/*
	 * Finds the set of angles at which the robot will collide with an obstacle
	 */
	std::vector<std::pair<std::pair<float, float>, AliceStructs::obj>> findDeadZones();



	/*
	 * Avoids other robots dynamically
	 */
	bool avoidNeighbors();

	/*
	 * Prevents Alice from hitting obstacles or other robots
	 */
	void avoidCollisions();

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
	AliceStructs::pnt findContour();

	/*
	 * Makes Alice go to a target
	 */
	AliceStructs::pnt goToTar();

	/*
	 * Makes Alice explore new territory
	 */
	AliceStructs::pnt explore();
	//=========================================================================================================================

};

#endif /* RULES_H */
