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

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <swarm_server/battery_states.h>
#include <wvu_swarm_std_msgs/chargers.h>
#include <wvu_swarm_std_msgs/priorities.h>
#include <wvu_swarm_std_msgs/energy.h>
#include <wvu_swarm_std_msgs/sensor_data.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_points.h>

#include "sim_settings.h"

class Hawk_Sim
{
private:
	float counter = 0;
	float prev_counter = 1;

	bool first = true;
	bool energy_first = true;
	bool new_chargers;
	std::vector<std::pair<float, wvu_swarm_std_msgs::vicon_point>> food_targets;//creates a vector with food locations and amounts
	//palce holders for subscription data ---------------
	wvu_swarm_std_msgs::chargers temp_chargers;
	wvu_swarm_std_msgs::chargers prev_temp_chargers;
	float prev_rid = -1;
	wvu_swarm_std_msgs::priorities temp_priorities;
	wvu_swarm_std_msgs::energy temp_energy;
	wvu_swarm_std_msgs::sensor_data temp_sd;
	wvu_swarm_std_msgs::vicon_bot_array temp_bots;
	ros::Time time;
	//---------------------------------------------------

	//callback functions
	void chargersCallback(const wvu_swarm_std_msgs::chargers &msg);
	void priorityCallback(const wvu_swarm_std_msgs::priorities &msg);
	void energyCallback(const wvu_swarm_std_msgs::energy &msg);
	void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg);

	//initializer functions
	void makeChargers(ros::Publisher _pub);
	void makePriority(ros::Publisher _pub);
	void makeSensorData(ros::Publisher _pub);
	void makeTargets(ros::Publisher _pub);


public:
	void run(ros::NodeHandle n);
};
