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

#include "Pvector.h"
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_bot.h>
#include <wvu_swarm_std_msgs/vicon_points.h>

#ifndef BODY_H_
#define BODY_H_

// The Body Class
//
// Attributes
//  bool predator: flag that specifies whether a given body is a predator.
//  Pvector location: Vector that specifies a bodies' location.
//  Pvector velocity: Vector that specifies a bodies' current velocity.
//  Pvector acceleration: Vector that specifies a bodies' current acceleration.
//  float maxSpeed: Limits magnitude of velocity vector.
//  float maxForce: Limits magnitude of acceleration vector. (F = m*a!)
//
// Methods
//  applyForce(Pvector force): Adds the given vector to acceleration
//
//  Pvector Separation(vector<Body> Bodies): If any other bodies are within a
//      given distance, Separation computes a a vector that distances the
//      current body from the bodies that are too close.
//  Pvector Alignment(vector<Body> Bodies): Computes a vector that causes the
//      velocity of the current body to match that of bodies that are nearby.
//
//  Pvector Cohesion(vector<Body> Bodies): Computes a vector that causes the
//      current body to seek the center of mass of nearby bodies.

class Body
{

private:
	float l; //distance between wheels
	bool applyForce(bool _aorb, float _rel_theta);

	//Physics helper functions.
	float angleConvert(float _x);
	int quadrant(float phi);
	bool aboveOrBelow(float _dx, float _dy, int _Q);
	float getRelTheta(float _abs_theta, float _phi, int _Q);

public:
	bool bodyPause;
	char id[2];
	int numid;
	wvu_swarm_std_msgs::vicon_points *targets;
	float heading;
	int sid;
	float force;
	float a; //left wheel speed
	float b; //right wheel speed
//	bool predator;
	ros::Time curTime;
	Pvector location;
	Pvector prev_location;
	Pvector velocity;
	Pvector acceleration;
	float maxSpeed;
	float maxForce;
//	bool updatedCommand;
//	bool updatedPosition;
	bool collision;

	Body()
	{
	}
	Body(float x, float y, char _id[2], int _numid);

	//Functions involving SFML and visualisation linking
	void run(vector<Body> v);
	void update();
	void borders();
	std::pair<float, float> borders(float _fx, float _fy);
	void inElasticCollisions(vector<Body> _bodies);
	float angle(Pvector v);
};

#endif
