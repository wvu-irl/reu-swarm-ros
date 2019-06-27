/*
 * Alice.h
 *
 * Serves as a manager for the model and rules by updating and calling them.
 * Not very functional; mainly exists for organizational reasons.
 *
 * Author: Casey Edmonds-Estes
 */

#ifndef ALICE_H
#define ALICE_H

#include "alice_swarm/aliceStructs.h"

class Alice
{

public:
	int name;

	/*
	 * Dummy constructor for the compiler
	 */
	Alice();

	/*
	 * Initializes Alice, rules, and the model
	 */
	Alice(AliceStructs::mail _data);

	/*
	 * Updates the model with new data
	 */
	void updateModel(AliceStructs::mail _data);

	/*
	 * Generates a velocity vector to be sent to the low-level controllers
	 */
	AliceStructs::vel generateVel();

private:

	Rules rules;
	Model model;
};

#endif /* ALICE_H */
