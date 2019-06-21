#ifndef RULES_H_
#define RULES_H_

#include "alice_swarm/aliceStructs.h"
#include "math.h"

static const float ROBOT_SIZE = 5; //Size of the robot, in cm

class Rules
{
public:

	bool should_ignore;

	Rules();

	AliceStructs::ideal addIdeals(AliceStructs::ideal i1, AliceStructs::ideal i2);


	AliceStructs::vel dummy1();

	AliceStructs::vel goToTarget(std::list<AliceStructs::obj> targets, float strength, float fov);


	AliceStructs::vel maintainSpacing(std::list <AliceStructs::neighbor> bots,float strength);

	AliceStructs::vel magnetAvoid(std::list <AliceStructs::neighbor> bots, float strength);

	AliceStructs::vel birdAvoid(std::list <AliceStructs::neighbor> bots, float strength, float fov);

	AliceStructs::vel avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength, float fov);
	//vel goToTarget(std::list <obs> targets, float tolerance); to implement later

private:

};

#endif /* RULES_H_ */
