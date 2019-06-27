/*
 * Model.h
 *
 * Stores Alice's data about her world and acts as her memory. Mainly acts as a vast data structure
 * for the rules to use. Updated by Alice.
 */

#ifndef MODEL_H
#define MODEL_H

#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Alice.h"
#include "alice_swarm/Rules.h"

class Model
{
public:

	/*
	 * Information about Alice's immediate environment
	 */
	AliceStructs::obstacles current_obstacles;
	//AliceStructs::heat_map current_heat_map; Need to figure this out with Jeongwoo

	/*
	 * Information Alice is storing about the wider world
	 */
	AliceStructs::obstacles memento_obstacles;
	//AliceStructs::heatMap memento_heat_map; See above

	Model(int _name);

	/*
	 * Updates the model from the sensors; called by Alice
	 */
	void sensorUpdate(AliceStructs::mail _toAdd)

	/*
	 * Passes data to neighbors
	 */
	void pass(); //Exact implementation TBD

	/*
	 * Forgets unneeded data
	 */
	void forget();

	/*
	 * Calculates the collision state, or what the robot needs to do to avoid obstacles
	 */
	string getCollisionState();

	/*
	 * Finds the nearest point on an obstacle
	 */
	int findLocalMin(AliceStructs::obstacle);

private:

	/*
	 * Information about Alice herself
	 */
	int name;
};

#endif /* MODEL_H */
