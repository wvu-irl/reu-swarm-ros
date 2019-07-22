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
		float l; //distance between wheels
		bool applyForce(bool _aorb,float _rel_theta);

		//Physics helper functions.
		float angleConvert(float _x);
		int quadrant(float phi);
		bool aboveOrBelow(float _dx, float _dy, int _Q);
		float getRelTheta(float _abs_theta, float _phi, int _Q);

public:
		bool bodyPause;
		char id [2];
		wvu_swarm_std_msgs::vicon_points *targets;
		float heading;
		int sid;
		float force;
		float a;//left wheel speed
		float b;//right wheel speed
    bool predator;
    int numid;
    ros::Time curTime;
    Pvector location;
    Pvector prev_location;
    Pvector velocity;
    Pvector acceleration;
    float maxSpeed;
    float maxForce;
    bool updatedCommand;
    bool updatedPosition;
    bool collision;


    Body() {}
    Body(float x, float y, char _id[2], int _numid);
 //   Body(float x, float y, bool predCheck);

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
    std::pair<float,float> borders(float _fx, float _fy);
    void seperation(vector<Body> _bodies);
    void inElasticCollisions(vector<Body> _bodies);
    void elasticCollisions(vector<Body> _bodies);
    void targetCollision(int i, float _t_sep);
    void targetInElastic(int i, float _t_sep);
    float targetSeperation(wvu_swarm_std_msgs::vicon_point);
    float angle(Pvector v);
    void printMessage(int i,wvu_swarm_std_msgs::vicon_bot_array _vb_array);
};

#endif
