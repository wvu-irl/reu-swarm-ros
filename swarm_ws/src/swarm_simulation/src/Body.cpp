#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <SFML/Graphics.hpp>
#include <swarm_simulation/Body.h>

//msg creation and formating includes
#include "geometry_msgs/TransformStamped.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include "tf/LinearMath/Matrix3x3.h"
#include <tf2/LinearMath/Quaternion.h>

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

Body::Body(float x, float y)
{
    acceleration = Pvector(0, 0);
    velocity = Pvector(rand()%3 - 2, rand()%3 - 2);
    location = Pvector(x, y);
    ID[0] = 'I';
    ID[1] = 'D';
    maxSpeed = 1.5;
    maxForce = 0.5;
}

Body::Body(float x, float y, bool predCheck)
{
    predator = predCheck;
    if (predCheck == true) {
        maxSpeed = 2.0;
        maxForce = 0.5;
        velocity = Pvector(rand()%3 - 1, rand()%3 - 1);
    } else {
        maxSpeed = 1.5;
        maxForce = 0.5;
        velocity = Pvector(rand()%3 - 2, rand()%3 - 2);
    }
    acceleration = Pvector(0, 0);
    location = Pvector(x, y);
}

// Adds force Pvector to current force Pvector
void Body::applyForce(Pvector force)
{
    acceleration.addVector(force);
}

// Separation
// Keeps bodies from getting too close to one another
Pvector Body::Separation(vector<Body> Bodies)
{
    // Distance of field of vision for separation between bodies
    float desiredseparation = 30;
    Pvector steer(0, 0);
    int count = 0;
    // For every body in the system, check if it's too close
    for (int i = 0; i < Bodies.size(); i++) {
        // Calculate distance from current body to body we're looking at
        float d = location.distance(Bodies[i].location);
        // If this is a fellow body and it's too close, move away from it
        if ((d > 0) && (d < desiredseparation)) {
            Pvector diff(0,0);
            diff = diff.subTwoVector(location, Bodies[i].location);
            diff.normalize();
            diff.divScalar(d);      // Weight by distance
            steer.addVector(diff);
            count++;
        }
        // If current body is a predator and the body we're looking at is also
        // a predator, then separate only slightly
        if ((d > 0) && (d < desiredseparation) && predator == true
            && Bodies[i].predator == true) {
            Pvector pred2pred(0, 0);
            pred2pred = pred2pred.subTwoVector(location, Bodies[i].location);
            pred2pred.normalize();
            pred2pred.divScalar(d);
            steer.addVector(pred2pred);
            count++;
        }
        // If current body is not a predator, but the body we're looking at is
        // a predator, then create a large separation Pvector
        else if ((d > 0) && (d < desiredseparation+70) && Bodies[i].predator == true) {
            Pvector pred(0, 0);
            pred = pred.subTwoVector(location, Bodies[i].location);
            pred.mulScalar(900);
            steer.addVector(pred);
            count++;
        }
    }
    // Adds average difference of location to acceleration
    if (count > 0)
        steer.divScalar((float)count);
    if (steer.magnitude() > 0) {
        // Steering = Desired - Velocity
        steer.normalize();
        steer.mulScalar(maxSpeed);
        steer.subVector(velocity);
        steer.limit(maxForce);
    }
    return steer;
}

// Alignment
// Calculates the average velocity of bodeis in the field of vision and
// manipulates the velocity of the current body in order to match it
Pvector Body::Alignment(vector<Body> Bodies)
{
    float neighbordist = 50; // Field of vision

    Pvector sum(0, 0);
    int count = 0;
    for (int i = 0; i < Bodies.size(); i++) {
        float d = location.distance(Bodies[i].location);
        if ((d > 0) && (d < neighbordist)) { // 0 < d < 50
            sum.addVector(Bodies[i].velocity);
            count++;
        }
    }
    // If there are bodies close enough for alignment...
    if (count > 0) {
        sum.divScalar((float)count);// Divide sum by the number of close bodies (average of velocity)
        sum.normalize();            // Turn sum into a unit vector, and
        sum.mulScalar(maxSpeed);    // Multiply by maxSpeed
        // Steer = Desired - Velocity
        Pvector steer;
        steer = steer.subTwoVector(sum, velocity); //sum = desired(average)
        steer.limit(maxForce);
        return steer;
    } else {
        Pvector temp(0, 0);
        return temp;
    }
}

// Cohesion
// Finds the average location of nearby bodies and manipulates the
// steering force to move in that direction.
Pvector Body::Cohesion(vector<Body> Bodies)
{
    float neighbordist = 50;
    Pvector sum(0, 0);
    int count = 0;
    for (int i = 0; i < Bodies.size(); i++) {
        float d = location.distance(Bodies[i].location);
        if ((d > 0) && (d < neighbordist)) {
            sum.addVector(Bodies[i].location);
            count++;
        }
    }
    if (count > 0) {
        sum.divScalar(count);
        return seek(sum);
    } else {
        Pvector temp(0,0);
        return temp;
    }
}

// Limits the maxSpeed, finds necessary steering force and
// normalizes vectors
Pvector Body::seek(Pvector v)
{
    Pvector desired;
    desired.subVector(v);  // A vector pointing from the location to the target
    // Normalize desired and scale to maximum speed
    desired.normalize();
    desired.mulScalar(maxSpeed);
    // Steering = Desired minus Velocity
    acceleration.subTwoVector(desired, velocity);
    acceleration.limit(maxForce);  // Limit to maximum steering force
    return acceleration;
}

// Modifies velocity, location, and resets acceleration with values that
// are given by the three laws.
void Body::update()
{
    //To make the slow down not as abrupt
    acceleration.mulScalar(0.3);
    // Update velocity
    velocity.addVector(acceleration);
    // Limit speed
    velocity.limit(maxSpeed);
    location.addVector(velocity);
    // Reset accelertion to 0 each cycle
    acceleration.mulScalar(0);
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
    Pvector sep = Separation(v);
    Pvector ali = Alignment(v);
    Pvector coh = Cohesion(v);
    // Arbitrarily weight these forces
    sep.mulScalar(1.5);
    ali.mulScalar(1.0); // Might need to alter weights for different characteristics
    coh.mulScalar(1.0);
    // Add the force vectors to acceleration
    applyForce(sep);
    applyForce(ali);
    applyForce(coh);
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

wvu_swarm_std_msgs::vicon_bot_array Body::createMessages(std::vector<Body> _flock)//vector<Body> _flock) //flock is a global.
{
	wvu_swarm_std_msgs::vicon_bot_array vb_array;
	for (int i = 0; i < _flock.size();i++)
	{
		wvu_swarm_std_msgs::vicon_bot this_bot;
		geometry_msgs::TransformStamped this_bot_msg;
		tf2::Quaternion q;
		Body cur = _flock.at(i); //current body being looked at.

		float mag = cur.velocity.magnitude();
		q.setRPY( 0, 0, cur.angle(cur.velocity) - M_PI_2);  // Create this quaternion from roll=0/pitch=0/yaw (in radians)
		q.normalize();

		//translational information
		this_bot_msg.transform.translation.x = cur.location.x;
		this_bot_msg.transform.translation.y = cur.location.y;
		this_bot_msg.transform.translation.z = 0;

		//rotational information
		this_bot_msg.transform.rotation.x = q.x();
		this_bot_msg.transform.rotation.y = q.y();
		this_bot_msg.transform.rotation.z = q.z();
		this_bot_msg.transform.rotation.w = q.w();

		//header information
		this_bot_msg.header.seq = 0;
		this_bot_msg.header.frame_id = "0";
		//set child frame dummy val.
		this_bot_msg.child_frame_id = "0";

		//create the vicon_bot.
		this_bot.botPose = this_bot_msg;
		this_bot.botId[0] = cur.ID[0];
		this_bot.botId[1] = cur.ID[1];

		//add to the vector list
		vb_array.poseVect.push_back(this_bot);
	}
	return vb_array;
}

