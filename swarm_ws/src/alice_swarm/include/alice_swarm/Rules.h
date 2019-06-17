#ifndef RULES_H_
#define RULES_H_

#include "alice_swarm/aliceStructs.h"
#include "math.h"

static const float ROBOT_SIZE = 7; //Size of the robot, in cm

class Rules
{
public:

	bool should_ignore;

	Rules();

	std::pair<float,float> addPolarVectors(std::pair<float,float> v1, std::pair<float,float> v2);


	AliceStructs::vel dummy1();

	AliceStructs::vel maintainSpacing(std::list <AliceStructs::neighbor> bots,float strength);

	AliceStructs::vel avoidNeighbors(std::list <AliceStructs::neighbor> bots, float tolerance);

	AliceStructs::vel avoidObstacles(std::list <AliceStructs::obj> obstacles, float tolerance);

	AliceStructs::vel panicAvoid(std::list <AliceStructs::neighbor> bots, float tolerance);
	//vel goToTarget(std::list <obs> targets, float tolerance); to implement later

private:

};

#endif /* RULES_H_ */
