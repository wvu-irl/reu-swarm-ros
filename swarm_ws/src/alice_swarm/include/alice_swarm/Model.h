/*
 * Model.h
 *
 * Stores Alice's data about her world and acts as her memory. Mainly acts as a vast data structure
 * for the rules to use. Updated by Alice.
 */

#ifndef MODEL_H
#define MODEL_H

#include "aliceStructs.h"

class Model
{
public:

	/*
	 * Information about Alice's immediate environment
	 */
	std::vector<AliceStructs::obj> current_obstacles;
	AliceStructs::pnt cur_pose;
	std::vector<AliceStructs::neighbor> neighbors;
	std::vector<AliceStructs::flow> flows;
	std::vector<AliceStructs::pnt> targets;
	AliceStructs::pnt goTo;

	/*
	 * Information Alice is storing about the wider world
	 */
	AliceStructs::obj archived_obstacles;
	std::vector<AliceStructs::pnt> archived_levels;

	Model(int _name);

	/*
	 * Updates the model from the sensors; called by Alice
	 */
	void sensorUpdate(AliceStructs::mail _toAdd);

	/*
	 * Passes data to neighbors
	 */
	void pass(); //Exact implementation TBD

	/*
	 * Forgets unneeded data
	 */
	void forget();

private:

	/*
	 * Information about Alice herself
	 */
	int name;
};

#endif /* MODEL_H */
