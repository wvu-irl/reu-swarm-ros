/*
 * Rules
 *
 * This class doesn't actually contain any information, other than the robot's name and the swarm to which it belongs.
 * Instead, it contains a series of rules methods, which are called by the model to generate and ideal vector.
 * For more info, see Model.cpp
 *
 * Authors: Casey Edmonds-Estes and Jeongwoo Seo
 */

#ifndef RULES_H_
#define RULES_H_

#include "alice_swarm/aliceStructs.h"
#include "math.h"

static const float ROBOT_SIZE = 5; //Size of the physical robot, in centimeters.

class Rules
{
public:

	bool should_ignore;

	int sid; //Swarm ID, or which swarm the robot belongs to.

	/*
	 * Dummy constructor for compiler
	 */
	Rules();

	Rules(int _sid);

	/*
	 * Performs weighted vector addition to add two ideal vectors together, then returns the result.
	 */
	AliceStructs::ideal addIdeals(AliceStructs::ideal i1, AliceStructs::ideal i2);

	/*
	 * Dummy rule for testing other parts of Alice
	 */
	AliceStructs::ideal dummy1();

	/*
	 * End of the helper and init methods. Everything under this line is a rule
	<=====================================================================================================================>
	 */

	/*
	 * Makes bots converge to be some distance away from the center of their neighbors.
	 */
	AliceStructs::ideal maintainSpacing(std::list<AliceStructs::neighbor> bots, float strength, float distance);

	/*
	 * Uses an electromagnetic-style "force" to prevents robots from getting too close.
	 */
	AliceStructs::ideal magnetAvoid(std::list <AliceStructs::neighbor> bots, float strength);

	/*
	 * Makes the robot avoid robots in front of it, to make flocking behavior more elegant
	 */
	AliceStructs::ideal birdAvoid(std::list <AliceStructs::neighbor> bots, float strength);

	/*
	 * Moves a robot towards a target
	 */
	AliceStructs::ideal goToTarget(std::list<AliceStructs::obj> targets, float strength);

	/*
	 * Makes the robots follow a flow, which is a vector with a position
	 */
	AliceStructs::ideal followFlow(std::list<AliceStructs::ideal> flows, float strength);

	/*
	 * Makes the robot avoid obstacles using an electromagnetic model.
	 */
	AliceStructs::ideal avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength);
};

#endif /* RULES_H_ */
