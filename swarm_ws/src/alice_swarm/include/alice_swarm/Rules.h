#ifndef RULES_H_
#define RULES_H_

#include "aliceStructs.h"
#include "math"

class Rules
{
public:
	vel maintainSpacing(std::list <neighbor> bots, float tolerance);

	vel avoidRobots(std::list <neighbor> bots, float tolerance);

	vel avoidObstacles(std::list <obs> obstacles, float tolerance);

	//vel goToTarget(std::list <obs> targets, float tolerance); to implement later

private:
	const float ROBOT_SIZE = 7; //Size of the robot, in cm

};

#endif /* RULES_H_ */
