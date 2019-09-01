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
 * Stores all the data types for Alice
 *
 * Author: Casey Edmonds-Estes
 */
#ifndef ALICESTRUCTS_H
#define ALICESTRUCTS_H

#include <string>
#include <vector>
#include <list>
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/chargers.h>
#include <swarm_server/battery_states.h>

namespace AliceStructs
{

struct vector_2f
{
	double x, y;
	double dx, dy;
	bool valid;
};

/*
 * A vector for Alice to follow
 */
typedef struct
{
	float x;
	float spd;
	float y;
	float pri;
	ros::Time time;
	std::vector<int> observer_names;
} flow;

/*
 * A velocity
 */
typedef struct
{
	float dir;
	float mag;
} vel;

/*
 * Another robot near Alice
 */
typedef struct
{
	float x;
	float y;
	float tar_x;
	float tar_y;
	float ang;
	int name;
	int sid;
} neighbor;

/*
 * An object, usually an obstacle to be avoided
 */
typedef struct
{
	float x_rad;
	float y_rad;
	float x_off;
	float y_off;
	float theta_offset;
	ros::Time time;
	std::vector<int> observer_names;
} obj;

/*
 * A point
 */
typedef struct
{
	float x;
	float y;
	float z;
	ros::Time time;
	std::vector<int> observer_names;
} pnt;

/*
 * A pose
 */
typedef struct
{
	float x;
	float y;
	float z;
	float heading;
	ros::Time time;
	std::vector<int> observer_names;
} pose;


typedef struct
{
	float x;
	float y;
	bool occupied;
} charger;

/*
 * A box of all these structs, for easy transport
 */
typedef struct
{
	std::vector<neighbor> neighbors;
	std::vector<obj> obstacles;
	std::vector<pnt> targets;
	std::vector<flow> flows;
	std::vector<wvu_swarm_std_msgs::charger> *abs_chargers; //pointer to absolute chargers
	std::vector<wvu_swarm_std_msgs::charger> rel_chargers; //copy of chargers with pos relative to each bot.
	std::vector<float> *priority; //{REST, CHARGE, CONTOUR, TARGET, EXPLORE}
	float xpos;
	float ypos;
	float contVal;
	float heading;
	int name;
	float vision;
	float battery_lvl;
	BATTERY_STATES battery_state;
	float energy;
	ros::Time time;
} mail;

};

#endif
