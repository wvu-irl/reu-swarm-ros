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


	AliceStructs::ideal dummy1();

	AliceStructs::ideal goToTarget(std::list<AliceStructs::obj> targets, float strength);

	AliceStructs::ideal followFlow(std::list<AliceStructs::ideal> flows, float strength);

	AliceStructs::ideal maintainSpacing(std::list <AliceStructs::neighbor> bots, float strength);

	AliceStructs::ideal magnetAvoid(std::list <AliceStructs::neighbor> bots, float strength);

	AliceStructs::ideal birdAvoid(std::list <AliceStructs::neighbor> bots, float strength);

	AliceStructs::ideal avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength);
	//ideal goToTarget(std::list <obs> targets, float tolerance); to implement later

private:

};

#endif /* RULES_H_ */
