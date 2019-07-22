/*
 * Model.h
 *
 * Stores Alice's data about her world and acts as her memory. Mainly acts as a vast data structure
 * for the rules to use. Updated by Alice.
 */

#ifndef MODEL_H
#define MODEL_H

#include "aliceStructs.h"
#include <wvu_swarm_std_msgs/map.h>

class Model
{
public:

	/*
	 * Constant values used mostly for path planning
	 */
	float MAX_LV;
	float MAX_AV;
	float SIZE = 5;
	float SAFE_DIS = 2;

	/*
	 * Information about Alice's immediate environment
	 */
	std::vector<AliceStructs::obj> obstacles;

	/*due to sensory limitations, we gave these two absolutes, but Alice should never directly use these,
	 * only should be using for things like velocity
	*/
	AliceStructs::pose cur_pose;
	AliceStructs::pose first_pose;

	std::vector<AliceStructs::pnt> neighbor_go_to; //stores the neighbors' goTo's in this robots' first frame.

	bool first;
	ros::Time time;
	float vision;
	float energy;

	//these variables allow for self charging.
	float prev_highest_i; //rule from last iteration with highest priority
	float battery_lvl; //in volts.
	float min_sep = 1000;
	BATTERY_STATES battery_state; // enumerator {ERROR = -2, NONE = -1, CHARGED, GOING, CHARGING}
	bool charge2 = false;
	bool charging;
	bool committed;
	int closest_pos;


	std::vector<AliceStructs::neighbor> neighbors;
	std::vector<AliceStructs::flow> flows;
	std::vector<AliceStructs::pnt> targets;
	std::vector<wvu_swarm_std_msgs::charger> *abs_chargers; //pointer to absolute chargers vector
	std::vector<wvu_swarm_std_msgs::charger> rel_chargers; //copy of chargers vector with pos in current frame, not absolute.
	std::vector<float> *priority;

	AliceStructs::pnt goTo;

	/*
	 * Information Alice is storing about the wider world
	 */

	//all of the archived and map objects are stored relative to the starting position
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
	 * This pair of functions transforms vectors to and from the current and original frame,
	 * and the last one transforms any given frame to the given robot's original frame.
	 */
	std::pair<float, float> transformCur(float _x, float _y);
	std::pair<float, float> transformFir(float _x, float _y);
	std::pair<float, float> transformFtF(float _x, float _y,float _ox, float _oy, float _oheading);

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
	void pass(ros::Publisher _pub);


	/*
	 * Takes neighbors' maps and places their data into the archives
	 */
	void receiveMap(std::vector<wvu_swarm_std_msgs::map> &_maps,  std::vector<int> &_ids);
	/*
	 * Forgets unneeded data
	 */
	void forget();
	void forgetObs(int TOLERANCE);
	void forgetTargets(int TOLERANCE);
	void forgetContour(int TOLERANCE);

private:

	/*
	 * Information about Alice herself
	 */
//	int name;
};

#endif /* MODEL_H */
