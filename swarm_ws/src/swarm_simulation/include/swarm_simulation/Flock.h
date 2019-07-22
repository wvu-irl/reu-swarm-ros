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
	  void printMessage(wvu_swarm_std_msgs::vicon_bot_array _vb_array);
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
