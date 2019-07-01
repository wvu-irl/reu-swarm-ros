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
#include "wvu_swarm_std_msgs/alice_mail.h"

class Alice
{
public:

	Rules rules;
	Model model;

	int name;

	Alice();

	/*
	 * Initializes Alice, rules, and the model
	 */
	Alice(wvu_swarm_std_msgs::alice_mail _data);

	/*
	 * Takes data from messages to structs
	 */
	AliceStructs::mail packageData(wvu_swarm_std_msgs::alice_mail _data);

	/*
	 * Updates the model with new data
	 */
	void updateModel(wvu_swarm_std_msgs::alice_mail _data);

	/*
	 * Generates a velocity vector to be sent to the low-level controllers
	 */
	AliceStructs::vel generateVel();

};

#endif /* ALICE_H */
