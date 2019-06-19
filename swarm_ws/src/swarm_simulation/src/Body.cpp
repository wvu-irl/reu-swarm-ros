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
    maxSpeed = 1;
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
    	//elasticCollisions(v);
    	//inElasticCollisions(v);
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

void Body::elasticCollisions(vector<Body> _bodies)
{
	    //Magnitude of separation between bodies
	    float desiredseparation = 24;
	    Pvector steer(0, 0);
	    int count = 0; //iterator

	    for (int i = 0; i < _bodies.size(); i++) // For every body in the system, check if it's too close
	    {
	    	collision = false;
	        // Calculate distance from current body to body we're looking at
	    	 float d = location.distance(_bodies.at(i).location);
	        // If this is a fellow body and it's too close, move away from it
	        if ((d <= desiredseparation) && !((id[0] ==_bodies.at(i).id[0]) && (id[1] ==_bodies.at(i).id[1]))) // (&&d>0))
	        {
	        	if(collision == false){collision = true;}
//	        	_bodies.at(i).location.x = _bodies.at(i).prev_location.x;
//	          location.x = prev_location.x;
//
//						_bodies.at(i).location.y = _bodies.at(i).prev_location.y;
//						location.y = prev_location.y;

	        	float fx = velocity.x;
	        	float fy = velocity.y;
	        	float fnx = -fx;
	      	  float fny = -fy;

	        	std::cout<<"---BOT-- "<<id[0]<<id[1]<<" With Bot: "<<_bodies.at(i).id[0]<<_bodies.at(i).id[1]<<std::endl;
	        	std::cout<<"forces (fx,fy) "<<fx<<","<<fy<<std::endl;
	        	std::cout<<"fn (fnx,fny) "<<fnx<<","<<fny<<std::endl;
	        	std::cout<<"heading"<<heading*180/M_PI<<std::endl;

	        	location.addVector(Pvector(fnx,fny));
	        	//_bodies.at(i).location.addVector(Pvector(-fnx,-fny));
	        }
	    }
	}



void  Body::inElasticCollisions(vector<Body> _bodies)
{
	    //Magnatude of separation between bodies
	    float desiredseparation = 24;
	    Pvector steer(0, 0);
	    int count = 0; //iterator

	    for (int i = 0; i < _bodies.size(); i++) // For every body in the system, check if it's too close
	    {
	    	collision = false;
	        // Calculate distance from current body to body we're looking at
	    	 float d = location.distance(_bodies.at(i).location);

	        // If this is a fellow body and it's too close, move away from it
	        if ((d <= desiredseparation) && !((id[0] ==_bodies.at(i).id[0]) && (id[1] ==_bodies.at(i).id[1]))) // (&&d>0))
	        {
	        	std::cout<<"---BOT-- "<<id[0]<<id[1]<<" With Bot: "<<_bodies.at(i).id[0]<<_bodies.at(i).id[1]<<std::endl;

	        	float fnx;
	        	float fny;

	        	float dy = _bodies.at(i).location.y - location.y;
					  float dx = _bodies.at(i).location.x - location.x;

					  std::cout<<"dx,dy: "<<dx<<","<<dy<<std::endl;

					  //angle b/w d and origin.
					  float phi = M_PI_2 + angle(Pvector(dy,dx));

				   	float fx = velocity.x;
						float fy = velocity.y;
						float abs_theta = angle(velocity); //global polar angle of velocity

						float rel_theta; //velocity direction in frame of the collision plane
//						float rel_theta = abs_theta - phi + M_PI_2 + M_PI; //velocity direction in frame of the collision plane

	        	if(collision == false)
	        	{
	        		collision = true;
	        	}

//	        	if(((abs_theta>0) && (abs_theta<=M_PI_2)) || ((abs_theta>M_PI) && (abs_theta<=3*M_PI_2)))//Q1 and Q3
	        	if(((phi>0) && (phi<=M_PI_2)) || ((phi>M_PI) && (phi<=3*M_PI_2)))//Q1 and Q3
	        	{
	        		rel_theta = abs_theta - phi - M_PI_2;
	        		std::cout<<"=Bot is in Q1 or 3=, phi is: "<<phi*180/M_PI<<std::endl;
	        		std::cout<<"*-velocity direction: "<<abs_theta*180/M_PI<<std::endl;

	        		if((dx>=0) && (dy>=0)) //above
	        		{
	        			std::cout<<"$-Bot is above-$"<<std::endl;
	        			std::cout<<"vel angle in rotated frame: "<<rel_theta*180/M_PI<<std::endl;

	        			if(((rel_theta>=M_PI_2)&&(rel_theta<=3*M_PI_2))||((rel_theta<=-M_PI_2)&&(rel_theta>=-3*M_PI_2)))//vector has any alignment with plane.
	        			{
	        				std::cout<<"directed towards the plane"<<std::endl;
	        				location.x = prev_location.x;
	        				location.y = prev_location.y;

	        				fnx = velocity.x - (fx*pow(sin(phi),2) + fy * sin(phi) * cos(phi));
	        			  fny = velocity.y - (fy*pow(cos(phi),2) + fx * sin(phi) * cos(phi));
	        			}
	        			else
	        			{
	        				std::cout<<"directed away from plane"<<std::endl;
	        				fnx = velocity.x;
	        				fny = velocity.y;
	        			}
	        		}
							else //bellow
							{
								std::cout<<"$-Bot is below-$"<<std::endl;
								std::cout<<"vel angle in rotated frame: "<<rel_theta*180/M_PI<<std::endl;
								if(((rel_theta >=0) && (rel_theta>M_PI_2)) || ((rel_theta<2*M_PI) && (rel_theta>3*M_PI_2))
										||((rel_theta<=0) && (rel_theta>=-M_PI_2))||((rel_theta<=-3*M_PI_2)&&(rel_theta>=-2*M_PI)))
									//vector has any alignment with plane.
								{
									location.x = prev_location.x;
									location.y = prev_location.y;

									fnx = velocity.x - (fx*pow(sin(phi),2) + fy * sin(phi) * cos(phi));
								  fny = velocity.y - (fy*pow(cos(phi),2) + fx * sin(phi) * cos(phi));
								}
								else
								{
									fnx = velocity.x;
									fny = velocity.y;
								}
							}
	        	}
//						else if(((abs_theta>M_PI_2) && (abs_theta<=M_PI)) || ((abs_theta>3*M_PI_2) && (abs_theta<=2*M_PI)))//Q2 and Q4
						else if(((phi>M_PI_2) && (phi<=M_PI)) || ((phi>3*M_PI_2) && (phi<=2*M_PI))||((phi<0)&&(phi>-3*M_PI_2)))//Q2 and Q4
						{
							rel_theta = abs_theta - phi + M_PI_2;
							std::cout<<"=Bot is in Q2 or 4=, phi is: "<<phi*180/M_PI<<std::endl;
							std::cout<<"*-velocity direction:"<<abs_theta*180/M_PI<<std::endl;
							if((dx<=0) && (dy>=0)) //above
							{
								std::cout<<"$-Bot is above-$"<<std::endl;
								std::cout<<"vel angle in rotated frame: "<<rel_theta*180/M_PI<<std::endl;
								if(((rel_theta>=M_PI_2)&&(rel_theta<=3*M_PI_2))||((rel_theta<=-M_PI_2)&&(rel_theta>=-3*M_PI_2)))//vector has any alignment with plane.
								{
									std::cout<<"directed towards the plane"<<std::endl;
									location.x = prev_location.x;
								  location.y = prev_location.y;

									fnx = velocity.x - (fx*pow(sin(phi),2) + fy * sin(phi) * cos(phi));
									fny = velocity.y - (fy*pow(cos(phi),2) + fx * sin(phi) * cos(phi));
								}
								else
								{
									std::cout<<"away from plane"<<std::endl;
									fnx = velocity.x;
									fny = velocity.y;
								}
							}
							else //bellow
							{
								std::cout<<"$-Bot is below-$"<<std::endl;
								std::cout<<"vel angle in rotated frame: "<<rel_theta*180/M_PI<<std::endl;
								if(((rel_theta >=0) && (rel_theta>M_PI_2)) || ((rel_theta<2*M_PI) && (rel_theta>3*M_PI_2))
										||((rel_theta < 0) && (rel_theta > -3*M_PI_2)))//vector has any alignment with plane.
								{
									std::cout<<"directed towards the plane"<<std::endl;
									location.x = prev_location.x;
									location.y = prev_location.y;

									fnx = velocity.x - (fx*pow(sin(phi),2) + fy * sin(phi) * cos(phi));
								  fny = velocity.y - (fy*pow(cos(phi),2) + fx * sin(phi) * cos(phi));
								}
								else
								{
									std::cout<<"away from plane"<<std::endl;
									location.x = prev_location.x;
									location.y = prev_location.y;

									fnx = velocity.x;
									fny = velocity.y;
								}
							}
						}
						else
						{
							std::cout<<"XXXXXXXXXXX--Fuck You Man--XXXXXXXXXXXXXXXXXXXXXX"<<std::endl;
							std::cout<<"phi is: "<<phi*180/M_PI<<std::endl;
						}

	        	std::cout<<"forc es (fx,fy) "<<fx<<","<<fy<<std::endl;
	        	std::cout<<"fn (fnx,fny) "<<fnx<<","<<fny<<std::endl;
	        	std::cout<<"heading: "<<heading*180/M_PI<<std::endl;
	        	std::cout<<"sep distance: "<<d<<std::endl;

	        	std::cout<<"location pre adjustment: "<<location.x<<","<<location.y<<std::endl;
	        	location.addVector(Pvector(fnx,fny));
	        	std::cout<<"adjusted location: "<<location.x<<","<<location.y<<std::endl;
	        	std::cout<<"----------------------------------------------------"<<std::endl;
	        	//_bodies.at(i).location.addVector(Pvector(-fnx,-fny));
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
