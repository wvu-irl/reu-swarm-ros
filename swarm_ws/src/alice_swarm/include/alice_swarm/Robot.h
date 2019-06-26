/*
 * Robot
 *
 * The robot class serves as the interface to Alice's brain. It calls and updates Model and VectorQueue to ultimately
 * generate a velocity vector, which is then sent the low-level controllers.
 *
 * Authors: Jeongwoo Seo and Casey Edmonds-Estes
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

	/*
	 * Dummy constructor, exists only for compiler reasons
	 */
	Robot();

	/*
	 *	Initializes the model and passes data from the sensors to the model
	 */
	Robot(AliceStructs::mail data);

	/*
	 * Passes data from the sensors to the sensors to the model
	 */
	void receiveMsg(AliceStructs::mail data);

	/*
	 * Generates an ideal vector using the model, then passes that vector to the VectorQueue
	 */
	AliceStructs::ideal generateIdeal();

	/*
	 * This method first passes ideal vectors from neighbors in its swarm to its VectorQueue, then
	 * uses the updated queue to generate a velocity vector to be passed to the low-level controls
	 */
	AliceStructs::vel generateComp(std::vector<AliceStructs::ideal> ideals);

private:

	VectorQueue vectorQueue;
	Model model;


	std::vector<AliceStructs::neighbor> neighbors;

};

#endif /* ROBOT_H_ */
