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
	std::vector<AliceStructs::obj> obstacles;
	AliceStructs::pose cur_pose;
	AliceStructs::pose first_pose;
	bool first;
	ros::Time time;
	std::vector<AliceStructs::neighbor> neighbors;
	std::vector<AliceStructs::flow> flows;
	std::vector<AliceStructs::pnt> targets;
	AliceStructs::pnt goTo;

	/*
	 * Information Alice is storing about the wider world
	 */

	std::vector<AliceStructs::obj> archived_obstacles;
	std::vector<AliceStructs::pnt> archived_targets;
	std::vector<AliceStructs::pose> archived_contour;

	Model();

	Model(int _name);

	/*
	 * Clears data about alice's immediate environment
	 */
	void clear();

	/*
	 * Stores items that the robot has seen but have gone out of range
	 */
	void archiveAdd(AliceStructs::mail &_toAdd);


	/*
	 * Updates the model from the sensors; called by Alice
	 */
	void sensorUpdate(AliceStructs::mail &_toAdd);

	/*
	 * Passes data to neighbors
	 */
	void pass(ros::Publisher _pub); //Exact implementation TBD

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
