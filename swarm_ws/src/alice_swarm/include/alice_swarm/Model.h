/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

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
	float MAX_LV = 5;
	float MAX_AV = 5;
	float SIZE = 10;
	float SAFE_DIS = 5;

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

public:

	/*
	 * Information about Alice herself
	 */
	int name;
};

#endif /* MODEL_H */
