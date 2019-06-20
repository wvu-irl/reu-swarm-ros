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
    maxSpeed = 0.9;
    maxForce = 0.5;
    id[0] = _id[0];
    id[1] = _id[1];
    heading = 0;
    force =1;
    updatedCommand = false;
    updatedPosition = false;
    collision = false;
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

//void Body::applyForce(Pvector force) //this is the old applyForce
//{
//    acceleration.addVector(force);
//}

float Body::angleConvert(float _x)
{
	if(_x<0)
	{
		_x = _x + 2*M_PI;
	}
	_x = fmod(_x,2*M_PI);
	return _x;
}

int Body::quadrant(float _phi)//accepts angle of plane, returns quadrant of that plane as int (1,2,3,4).
{
	int Q;
	if((_phi>=0)&&(_phi<M_PI_2))
	{
		Q = 1;
	}
	else if((_phi>=M_PI_2)&&(_phi<=M_PI))
	{
		Q = 2;
	}
	else if((_phi>M_PI)&&(_phi<3*M_PI_2))
	{
		Q = 3;
	}
	else
	{
		Q = 4;
	}
	std::cout<<"--IN Quadrant: "<<Q<<"--"<<std::endl;
	return Q;
}

bool Body::aboveOrBelow(float _dx, float _dy, int _Q)//returns whether bot is above or below collision plane.
{
	bool place = false; //above is true, below is false
	if((_Q == 1)||(_Q == 3))
	{
		if((_dx>=0)&&(_dy>=0))
		{
			std::cout<<"&- Bot is above the plane -&"<<std::endl;
			place = true;
		}
		else{std::cout<<"&- Bot is below the plane -&"<<std::endl;}
	}
	else if((_Q == 2)||(_Q == 4))
	{
		if((_dx<=0)&&(_dy>=0))
		{
			std::cout<<"&- Bot is above the slope -&"<<std::endl;
			place = true;
		}
		else{std::cout<<"&- Bot is below the plane -&"<<std::endl;}
	}
	return place;
}

float Body::getRelTheta(float _abs_theta, float _phi, int _Q)
{
	float rel_theta;
	if((_Q ==1)||(_Q==4))
	{
		rel_theta = _abs_theta - _phi - M_PI_2;
	}
	else if((_Q==2)||(_Q==3))
	{
		rel_theta = _abs_theta - _phi + M_PI_2;
	}
	else
	{
		std::cout<<"the program broke getRelTheta()"<<std::endl;
	}
	rel_theta = angleConvert(rel_theta); //scales anlge to be positive.
	return rel_theta;
}

bool Body::applyForce(bool _aorb,float _rel_theta)
{
	bool apply = false;
	if(_aorb==true) //above the plane
	{
		if((_rel_theta > M_PI_2) && (_rel_theta < 3*M_PI_2))
		{
			std::cout<<"will apply force"<<std::endl;
			apply = true;
		}
		else{std::cout<<"will not apply force"<<std::endl;}
	}
	else if(_aorb == false) //bot is below plane
	{
		if(((_rel_theta >=0) && (_rel_theta < M_PI_2)) || ((_rel_theta> 3*M_PI_2) && (_rel_theta < 2*M_PI)))
		{
			std::cout<<"will apply force"<<std::endl;
			apply = true;
		}
		else{std::cout<<"will not apply force"<<std::endl;}
	}
	else
	{
		std::cout<<"shit is broke in applyForce()"<<std::endl;
	}
	return apply;
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
    	//elasticCollisions(v);
    	inElasticCollisions(v);
    	//seperation(v);
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
    	//velocity.x = 0;
    	if(location.x <= 12){location.x = 12;}
    	if(location.x >= 288){location.x = 288;}
    }
    if ((location.y <=12) ||(location.y >=588))
		{
			//velocity.y = 0;
			if(location.y <=12){location.y = 12;}
			if(location.y >=588){location.y = 588;}
		}
}

void Body::inelasticCollisions(vector<Body> _bodies)
{
	    //Magnitude of separation between bodies
	    float desiredseparation = 15;
	    for (int i = 0; i < _bodies.size(); i++) // For every body in the system, check if it's too close
	    {
	    	collision = false;
	        // Calculate distance from current body to body we're looking at
	    	 float d = location.distance(_bodies.at(i).location);
	        // If this is a fellow body and it's too close, move away from it
	        if ((d <= desiredseparation) && !((id[0] ==_bodies.at(i).id[0]) && (id[1] ==_bodies.at(i).id[1]))) // (&&d>0))
	        {
	        	if(collision == false)
	        	{
	        		collision = true;
	        	}

	        	std::cout<<"---BOT-- "<<id[0]<<id[1]<<" With Bot: "<<_bodies.at(i).id[0]<<_bodies.at(i).id[1]<<std::endl;

						float fnx; //force to be applied (x).
						float fny; //force to be applied (y).

						float dy = _bodies.at(i).location.y - location.y;
						float dx = _bodies.at(i).location.x - location.x;

						std::cout<<"dx,dy: "<<dx<<","<<dy<<std::endl;

						//angle b/w d and origin.
						float d_angle = angle(Pvector(dx,dy));
						d_angle = angleConvert(d_angle);

						float phi = M_PI_2 + d_angle;
						phi = angleConvert(phi);

						std::cout<<"Angle of seperation: "<<d_angle*180/M_PI<<std::endl;
						std::cout<<"Angle of collision: "<<phi*180/M_PI<<std::endl;
//				   	float fx = velocity.x;
//						float fy = velocity.y;
//						float abs_theta = angle(velocity); //global polar angle of velocity

						int h = quadrant(heading);
						float fx = force*cos(heading); //bot force in heading direction(x).
						float fy = abs(force*sin(heading)); //bot force in heading direction (y).

						if((h == 1)||(h==2))
						{
							fy = -fy;
						}

						float mag = sqrt(pow(fx,2) + pow(fy,2));
						fx = fx/mag; //scaled to unit vectors
						fy = fy/mag;

						float abs_theta = angleConvert(angle(Pvector(fx,fy))); //angle of force in absolute position.
						abs_theta = angleConvert(abs_theta);

						int Q = quadrant(phi); //quadrant bot is in
						bool above_below = aboveOrBelow(dx,dy,Q); //gets quadrant collision is happening.
						float rel_theta = getRelTheta(abs_theta,phi,Q); //velocity direction in frame of the collision plane.
						bool apply_force = applyForce(above_below, rel_theta); //finds if force points towards the plane.

						if(apply_force == true)
						{
							location.x = prev_location.x;
							location.y = prev_location.y;
							fnx = fx - (fx*pow(sin(phi),2) + fy * sin(phi) * cos(phi));
							fny = fy - (fy*pow(cos(phi),2) + fx * sin(phi) * cos(phi));
						}
						else
						{
							location.x = prev_location.x;
							location.y = prev_location.y;
							fnx = fx;
							fny = fy;
						}
						std::cout<<"forces direction: "<<abs_theta*180/M_PI<<std::endl;
						std::cout<<"forces (fx,fy) "<<fx<<","<<fy<<std::endl;
						std::cout<<"fn (fnx,fny) "<<fnx<<","<<fny<<std::endl;
						std::cout<<"heading: "<<heading*180/M_PI<<std::endl;
						std::cout<<"relative theta: "<<rel_theta*180/M_PI<<std::endl;
						std::cout<<"sep distance: "<<d<<std::endl;
						std::cout<<"location pre adjustment: "<<location.x<<","<<location.y<<std::endl;

						location.addVector(Pvector(fnx,fny));//IMPORTANT

						std::cout<<"adjusted location: "<<location.x<<","<<location.y<<std::endl;
						std::cout<<"----------------------------------------------------"<<std::endl;
	        }
	    }
}

void  Body::seperation(vector<Body> _bodies)
{
		// Distance of field of vision for separation between bodies
    float desiredseparation = 15;
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
//        	std::cout<<"-----------------\n";
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
