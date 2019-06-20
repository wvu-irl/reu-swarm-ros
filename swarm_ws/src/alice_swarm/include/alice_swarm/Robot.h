/*
 * Robot.h
 *
 *  Created on: Jun 13, 2019
 *      Author: smart6
 */

#ifndef ROBOT_H_
#define ROBOT_H_

#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Model.h"
#include "alice_swarm/VectorQueue.h"

class Robot {

public:
	int name;
	int sid;

	VectorQueue vectorQueue;

	Robot();

	Robot(AliceStructs::mail data);

	AliceStructs::ideal generateIdeal();

	AliceStructs::vel generateComp(std::vector <AliceStructs::ideal> ideals);

	void receiveMsg(AliceStructs::mail data);

private:
	Model model;

	std::vector <AliceStructs::neighbor> neighbors;

};

#endif /* ROBOT_H_ */
