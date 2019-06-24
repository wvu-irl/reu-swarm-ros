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

class Robot
{

public:
	int name;
	int sid;

	VectorQueue vectorQueue;

	Robot(); //dummy constructor

	Robot(AliceStructs::mail data); //actual constructor

	AliceStructs::ideal generateIdeal(); // generates ideal from the env. information given

	AliceStructs::vel generateComp(std::vector<AliceStructs::ideal> ideals); //generates a vector to go to after accounting for its neighbor's ideals

	void receiveMsg(AliceStructs::mail data); //takes in the mail struct for processing

private:
	Model model; //stores model, which contains info about its surroundings

	std::vector<AliceStructs::neighbor> neighbors; //stores the robot's neighbors

};

#endif /* ROBOT_H_ */
