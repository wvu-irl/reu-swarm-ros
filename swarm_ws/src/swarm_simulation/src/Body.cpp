#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <SFML/Graphics.hpp>
#include <swarm_simulation/Body.h>


// Global Variables for borders()
// desktopTemp gets screen resolution of PC running the program
sf::VideoMode desktopTemp = sf::VideoMode::getDesktopMode();
const int window_height = desktopTemp.height;
const int window_width = desktopTemp.width;

#define w_height window_height
#define w_width window_width
#define PI 3.141592635

// Body Functions from Body.h
// ----------------------------

Body::Body(float x, float y, char _id[2])
{
    acceleration = Pvector(0, 0);
    velocity = Pvector(0,0);
    location = Pvector(x, y);
    maxSpeed = 1;
    maxForce = 0.5;
    id[0] = _id[0];
    id[1] = _id[1];
}

//Body::Body(float x, float y, bool predCheck)
//{
//    predator = predCheck;
//    if (predCheck == true) {
//        maxSpeed = 2.0;
//        maxForce = 0.5;
//        velocity = Pvector(rand()%3 - 1, rand()%3 - 1);
//    } else {
//        maxSpeed = 1.5;
//        maxForce = 0.5;
//        velocity = Pvector(rand()%3 - 2, rand()%3 - 2);
//    }
//    acceleration = Pvector(0, 0);
//    location = Pvector(x, y);
//}

// Adds force Pvector to current force Pvector
void Body::applyForce(Pvector force)
{
    acceleration.addVector(force);
}


// Modifies velocity, location, and resets acceleration with values that
// are given by the three laws.
void Body::update()
{
    //To make the slow down not as abrupt
    // Update velocity
    //velocity.addVector(acceleration);
    // Limit speed
    velocity.limit(maxSpeed);
    location.addVector(velocity);
    // Reset accelertion to 0 each cycle

}

// Run flock() on the flock of bodies.
// This applies the three rules, modifies velocities accordingly, updates data,
// and corrects bodies which are sitting outside of the SFML window
void Body::run(vector <Body> v)
{
    flock(v);
    update();
    borders();
}

// Applies the three laws to the flock of bodies
void Body::flock(vector<Body> v)
{
//    Pvector sep = Separation(v);
//    Pvector ali = Alignment(v);
//    Pvector coh = Cohesion(v);
//    // Arbitrarily weight these forces
//    sep.mulScalar(1.5);
//    ali.mulScalar(1.0); // Might need to alter weights for different characteristics
//    coh.mulScalar(1.0);
//    // Add the force vectors to acceleration
//    applyForce(sep);
//    applyForce(ali);
//    applyForce(coh);
}

// Checks if bodies go out of the window and if so, wraps them around to
// the other side.
void Body::borders()
{
    if (location.x < 0) location.x += w_width;
    if (location.y < 0) location.y += w_height;
    if (location.x > 1000) location.x -= w_width;
    if (location.y > 1000) location.y -= w_height;
}

// Calculates the angle for the velocity of a body which allows the visual
// image to rotate in the direction that it is going in.
float Body::angle(Pvector v)
{
    // From the definition of the dot product
    float angle = (float)(atan2(v.x, -v.y) * 180 / PI);
    return angle;
}
//char[2] Body::getID()
//{
//	return ID;
//}
