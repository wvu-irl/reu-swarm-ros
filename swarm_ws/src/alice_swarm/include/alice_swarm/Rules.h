#ifndef RULES_H_
#define RULES_H_

#include "alice_swarm/aliceStructs.h"
#include "math.h"

static 	const float ROBOT_SIZE = 7; //Size of the robot, in cm

class Rules
{
public:

	Rules();

	AliceStructs::vel maintainSpacing(std::list <AliceStructs::neighbor> bots, float tolerance);

	AliceStructs::vel avoidRobots(std::list <AliceStructs::neighbor> bots, float tolerance);

	AliceStructs::vel avoidObstacles(std::list <AliceStructs::obj> obstacles, float tolerance);


	//vel goToTarget(std::list <obs> targets, float tolerance); to implement later

private:

};

#endif /* RULES_H_ */
