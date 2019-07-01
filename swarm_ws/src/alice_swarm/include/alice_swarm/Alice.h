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

#include "aliceStructs.h"
#include "Model.h"
#include "Rules.h"

class Alice
{
private:

	Rules rules;
	Model model;

public:
	int name;
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

};

#endif /* ALICE_H */
