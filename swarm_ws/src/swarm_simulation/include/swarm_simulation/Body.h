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

class Body {

private:

	  //wvu_swarm_std_msgs::vicon_bot_array createMessages(std::vector<Body> _flock);//vector<Body> _flock); //operates on flock
		//char[2] getID();
public:
		char id [2];
    bool predator;
    Pvector location;
    Pvector velocity;
    Pvector acceleration;
    float maxSpeed;
    float maxForce;
    bool updated;


    Body() {}
    Body(float x, float y, char _id[2]);
    Body(float x, float y, bool predCheck);
    void applyForce(Pvector force);
    // Three Laws that bodies follow
    Pvector Separation(vector<Body> Bodies);
    Pvector Alignment(vector<Body> Bodies);
    Pvector Cohesion(vector<Body> Bodies);
    //Functions involving SFML and visualisation linking
    Pvector seek(Pvector v);
    void run(vector <Body> v);
    void update();
    void flock(vector <Body> v);
    void borders();
    float angle(Pvector v);
    void printMessage(int i,wvu_swarm_std_msgs::vicon_bot_array _vb_array);
};

#endif
