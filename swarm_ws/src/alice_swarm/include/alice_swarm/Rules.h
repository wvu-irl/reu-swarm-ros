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

	vel goToTarget(std::list <tar> targets, float tolerance);

private:
	const float ROBOT_SIZE = 7; //Size of the robot, in cm
	const int SENS = 1; //Turn up if computer can handle it to make Alice smarter

};

#endif /* RULES_H_ */
