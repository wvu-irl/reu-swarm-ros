#include "Pvector.h"
#include <vector>
#include <stdlib.h>
#include <iostream>

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
public:
    bool predator;
    Pvector location;
    Pvector velocity;
    Pvector acceleration;
    float maxSpeed;
    float maxForce;
    Body() {}
    Body(float x, float y);
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
};

#endif
