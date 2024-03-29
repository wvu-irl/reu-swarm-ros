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

#ifndef HUB_H
#define HUB_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/flows.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <wvu_swarm_std_msgs/chargers.h>
#include <wvu_swarm_std_msgs/charger.h>
#include <wvu_swarm_std_msgs/energy.h>
#include <wvu_swarm_std_msgs/sensor_data.h>

#include "alice_swarm/aliceStructs.h"
#include <contour_node/level_description.h>

typedef struct Bot //The Bot struct holds the pose of a robot on the absolute frame, along with its distance from another.
{
	Bot() //Default Constructor
	{
		id = -1;
		x = 0;
		y = 0;
		heading = 0;
		distance = 10000;
		swarm_id = -1; //deprecated
	}

	Bot(int _id, float _x, float _y, float _heading, float _distance, int _sid, ros::Time _time) //Alternate Constructor
	{
		id = _id;

		x = _x;
		y = _y;
		heading = _heading;
		distance = _distance;
		swarm_id = _sid;
		time = _time;
	}

	int id; //the id's are the 50 states, from 0 to 49
	int swarm_id; //which swarm it's in
	float x; //x position
	float y; //y position
	float heading; //in radians
	float distance; //
	ros::Time time; //timestamp of the observation
} Bot;

static const int NEIGHBOR_COUNT = 4; // Maximum number of neighbors for each robot
static const int VISION = 30; //Distance the robots can see

/*
 *Acts as a parser for the data coming out of the VICON or the simulation, turning the bundle of data into 50 packets
 *of individualized information for each swarm bot, allowing them to enact their agent level rules.
 */
class Hub
{

private:
	wvu_swarm_std_msgs::vicon_bot_array viconBotArray;
	wvu_swarm_std_msgs::vicon_points targets;
	wvu_swarm_std_msgs::map_levels map;
	wvu_swarm_std_msgs::flows flows;
	wvu_swarm_std_msgs::chargers chargers;
	wvu_swarm_std_msgs::energy energy;
	wvu_swarm_std_msgs::sensor_data sensor_data;
	std::vector<wvu_swarm_std_msgs::sensor_data> sensor_datas;

	std::vector<Bot> bots; //holds locations of all of the bots

	std::vector<std::vector<Bot>> neighbors; //holds locations of all of the neighboring bots relative to each other

	std::pair<float, float> getSeparation(Bot _bot, std::pair<float, float> _obs); 	//Finds the distance between a bot and some object

	void processVicon(); //Fills in bots[], converst vicon to just the pose data needed

	void updateSensorDatas(); //updates the sensor_datas vector by adding new elements, replacing old, and keeping order.

	bool validRID(); //checks the sensor data comes from a robot currently in operation

	void findNeighbors(); // Finds each robot's nearest neighbors, and thus fills out the neighbors vector

	void addFlowMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the flows within a robot's VISION range

	void addObsMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the obstacles within a robot's VISION range

	void addTargetMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the targets within a robot's VISION range

	void addNeighborMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the neighbors determined by its closest x

	void addContMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Gives each robot it's value on the contour map

	void addChargerMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Gives each robot charger station info
public:
	std::vector<int> ridOrder; //holds the order of the id's of the bots

	Hub(int a); //Default constructor, dummy parameter is there for compile reasons?

	//Adds the msgs gather from various topics to the private fields of Hub
	void update(wvu_swarm_std_msgs::vicon_bot_array &_b, wvu_swarm_std_msgs::vicon_points &_t,
			wvu_swarm_std_msgs::map_levels &_o, wvu_swarm_std_msgs::flows &_f, wvu_swarm_std_msgs::chargers &_c,
			wvu_swarm_std_msgs::energy &_e, wvu_swarm_std_msgs::sensor_data &_sd);

	wvu_swarm_std_msgs::alice_mail_array getAliceMail(); //Gathers all the relative information for a robot into one msg

	void clearHub(); //Clears neighbor information
};
#endif
