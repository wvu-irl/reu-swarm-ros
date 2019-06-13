/*
 * Robot.h
 *
 *  Created on: Jun 13, 2019
 *      Author: smart6
 */

#ifndef ROBOT_H_
#define ROBOT_H_

#include "aliceStructs.h"
#include "Model.h"
#include "VectorQueue.h"

class Robot {
private:
	Model model;
	VectorQueue vector_queue;
	std::vector <neighbor> neighbors;
public:
	int name;

	Robot(mail data);

	ideal generateIdeal();

	vel generateComp(std::vector <ideal> ideals);

};

#endif /* ROBOT_H_ */
