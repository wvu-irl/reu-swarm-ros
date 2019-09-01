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

#include <iostream>
#include <vector>
#include "Body.h"
#include "math.h"

#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_bot.h>
#include <wvu_swarm_std_msgs/vicon_points.h>

//msg creation and formating includes
#include "geometry_msgs/TransformStamped.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include "tf/LinearMath/Matrix3x3.h"
#include <tf2/LinearMath/Quaternion.h>

#ifndef FLOCK_H_
#define FLOCK_H_


// This file contains the class needed to create a flock of bodies. It utilizes
// the bodies class and initializes body flocks with parameters that can be
// specified. This class will be utilized in main.

class Flock {
public:
	  wvu_swarm_std_msgs::vicon_bot_array createMessages(wvu_swarm_std_msgs::vicon_bot_array);//vector<Body> _flock); //operates on flock
	  vector<Body> bodies;
    //Constructors
    Flock() {}

    // Accessor functions
    int getSize();
    Body getBody(int i);

    // Mutator Functions
    void addBody(Body b);
    void applyPhysics(wvu_swarm_std_msgs::vicon_points *_targets);
};

#endif
