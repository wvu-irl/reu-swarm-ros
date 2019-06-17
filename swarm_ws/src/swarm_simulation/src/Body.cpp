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

#define w_height 600
#define w_width 300
#define PI 3.141592635

// Body Functions from Body.h
// ----------------------------

Body::Body(float x, float y, char _id[2])
{
    acceleration = Pvector(0, 0);
    velocity = Pvector(0,0);
    location = Pvector(x, y);
    prev_location = Pvector(x,y);
    maxSpeed = 3;
    maxForce = 0.5;
    id[0] = _id[0];
    id[1] = _id[1];
    updatedCommand = false;
    updatedPosition = false;
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
    velocity.limit(1);
    velocity.mulScalar(maxSpeed);
    prev_location.set(location.x,location.y);
    location.addVector(velocity);

    // Reset accelertion to 0 each cycle
    updatedPosition=true;
}

// Run flock() on the flock of bodies.
// This applies the three rules, modifies velocities accordingly, updates data,
// and corrects bodies which are sitting outside of the SFML window
void Body::run(vector <Body> v)
{
    if (updatedPosition==false)
    {
    	update();
    	seperation(v);
    	borders();
    }


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
		//code for boundary wrapping
    /*if (location.x < 0) location.x += w_width;
    if (location.y < 0) location.y += w_height;
    if (location.x > 300) location.x -= w_width;
    if (location.y > 600) location.y -= w_height;*/

		//code for hard boundary conditions. Nulls velocity component orthogonal to boundary.
    if ((location.x <=12) ||(location.x >=288))
    {
    	velocity.x = 0;
    	if(location.x <= 12){location.x = 12;}
    	if(location.x >= 288){location.x = 288;}
    }
    if ((location.y <=12) ||(location.y >=588))
		{
			velocity.y = 0;
			if(location.y <=12){location.y = 12;}
			if(location.y >=588){location.y = 588;}
		}
}

void  Body::seperation(vector<Body> _bodies)
{
		// Distance of field of vision for separation between bodies
    float desiredseparation = 24;
    Pvector steer(0, 0);
    int count = 0; //iterator

    for (int i = 0; i < _bodies.size(); i++) // For every body in the system, check if it's too close
    {
        // Calculate distance from current body to body we're looking at
    	 float d = location.distance(_bodies.at(i).location);
        // If this is a fellow body and it's too close, move away from it
        if ((d < desiredseparation) && !((id[0] ==_bodies.at(i).id[0]) && (id[1] ==_bodies.at(i).id[1]))) // (&&d>0))
        {_bodies.at(i).location.x = _bodies.at(i).prev_location.x;
//        	std::cout<<"BOTS ARE TOO CLOSE!"<<std::endl;
//        	std::cout<<"bot: "<<id[0]<<id[1]<<" and bot: "<<_bodies.at(i).id[0]<<_bodies.at(i).id[1]<<std::endl;
//        	std::cout<<"seperation is: "<<d<<std::endl;
        	std::cout<<"-----------------\n";
        	_bodies.at(i).location.x = _bodies.at(i).prev_location.x;
        	location.x = prev_location.x;

        	_bodies.at(i).location.y = _bodies.at(i).prev_location.y;
        	location.y = prev_location.y;

//        	float dx = _bodies.at(i).location.x - location.x;
//        	float dy = _bodies.at(i).location.y - location.y;
//        	if(dx>=0)
//        	{
//        		_bodies.at(i).location.x += 30;
//        		location.x -= 30;
//        	}
//        	else
//        	{
//        		_bodies.at(i).location.x -= 30;
//        		location.x += 30;
//        	}
//        	if(dy>=0)
//        	{
//        		_bodies.at(i).location.y += 30;
//        		location.y -= 30;
//        	}
//        	else
//        	{
//        		_bodies.at(i).location.y -= 30;
//        		location.y += 30;
//        	}

//        	velocity.x = 0;
//          velocity.y = 0;
//          _bodies.at(i).velocity.x = 0;
//					_bodies.at(i).velocity.y =0;


//            Pvector diff(0,0);
//            diff = diff.subTwoVector(location, _bodies.at(i).location);
//            diff.normalize();
//            diff.divScalar(d);      // Weight by distance
//            steer.addVector(diff);
            count++;
        }
    }
}

// Calculates the angle for the velocity of a body which allows the visual
// image to rotate in the direction that it is going in.
float Body::angle(Pvector v)
{
    // From the definition of the dot product. negated to transform to first quadrant from 4th.
	  float angle = -1*(float)(atan2(v.y,v.x) );

    //float angle = (float)(atan2(v.x, -v.y) );
	  //^ the way this was written before. Saved it just in case.
    return angle;
}
