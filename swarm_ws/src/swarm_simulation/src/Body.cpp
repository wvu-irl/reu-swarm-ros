#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <SFML/Graphics.hpp>
#include <swarm_simulation/Body.h>
#include <swarm_simulation/sim_settings.h>

// Global Variables for borders()
// desktopTemp gets screen resolution of PC running the program
sf::VideoMode desktopTemp = sf::VideoMode::getDesktopMode();
const int window_height = desktopTemp.height;
const int window_width = desktopTemp.width;

#define BOD_DEBUG 0

// Body Functions from Body.h
// ----------------------------

Body::Body(float x, float y, char _id[2], int _numid) //constructor for each of the bodies (represent bots).
{
	numid = _numid;
	acceleration = Pvector(0, 0);
	velocity = Pvector(0, 0);
	location = Pvector(x, y);
	prev_location = Pvector(x, y);
	a = 0;
	b = 0;
	bodyPause = false;
	maxForce = 0.5;
	id[0] = _id[0];
	id[1] = _id[1];
	ros::Time curTime = ros::Time::now();
	heading = 0;
	force = 1;
	l = 10.5; //distance between wheels of robot
	collision = false;
}

//--------Helper Functions for the Physics engine-----------
float Body::angleConvert(float _x) //scales all angles to vals [0,2*pi].
{
	if (_x < 0)
	{
		_x = _x + 2 * M_PI;
	}
	_x = fmod(_x, 2 * M_PI);
	return _x;
}

int Body::quadrant(float _phi) //accepts angle of plane, returns quadrant of that plane as int (1,2,3,4).
{
	int Q;
	if ((_phi >= 0) && (_phi < M_PI_2))
	{
		Q = 1;
	}
	else if ((_phi >= M_PI_2) && (_phi <= M_PI))
	{
		Q = 2;
	}
	else if ((_phi > M_PI) && (_phi < 3 * M_PI_2))
	{
		Q = 3;
	}
	else
	{
		Q = 4;
	}
#if BOD_DEBUG
	std::cout<<"--IN Quadrant: "<<Q<<"--"<<std::endl;
#endif
	return Q;
}

bool Body::aboveOrBelow(float _dx, float _dy, int _Q) //returns whether bot is above or below collision plane.
{
	bool place = false; //above is true, below is false
	if ((_Q == 1) || (_Q == 3))
	{
		if ((_dx >= 0) && (_dy >= 0))
		{
#if BOD_DEBUG
			std::cout<<"&- Bot is above the plane -&"<<std::endl;
#endif
			place = true;
		}
		else
		{
#if BOD_DEBUG
			std::cout<<"&- Bot is below the plane -&"<<std::endl;
#endif
		}
	}
	else if ((_Q == 2) || (_Q == 4))
	{
		if ((_dx <= 0) && (_dy >= 0))
		{
#if BOD_DEBUG
				std::cout<<"&- Bot is above the slope -&"<<std::endl;
#endif
			place = true;

		}
		else
		{
#if BOD_DEBUG
				std::cout<<"&- Bot is below the plane -&"<<std::endl;
#endif
		}
	}
	return place;
}

float Body::getRelTheta(float _abs_theta, float _phi, int _Q) //converts global theta into frame of collision plane, with collision plane
{                                                        //as the vertical axis.
	float rel_theta;
	if ((_Q == 1) || (_Q == 4))
	{
		rel_theta = _abs_theta - _phi - M_PI_2;
	}
	else if ((_Q == 2) || (_Q == 3))
	{
		rel_theta = _abs_theta - _phi + M_PI_2;
	}
	else
	{
#if BOD_DEBUG
		std::cout<<"the program broke getRelTheta()"<<std::endl;
#endif
	}
	if (rel_theta == 3 * M_PI_2)
	{
		rel_theta = M_PI_2;
	}
	rel_theta = angleConvert(rel_theta); //scales anlge to be positive.
	return rel_theta;
}

bool Body::applyForce(bool _aorb, float _rel_theta) //decides whether a bot is going towards or away from the plane. Decides if force
{                                                    //needs to be applied.
	bool apply = false;
	if (_aorb == true)                                           //above the plane
	{
		if ((_rel_theta > M_PI_2) && (_rel_theta < 3 * M_PI_2))
		{
#if BOD_DEBUG
			std::cout<<"will apply force"<<std::endl;
#endif
			apply = true;
		}
		else
		{
#if BOD_DEBUG
			std::cout<<"will not apply force"<<std::endl;
#endif
		}
	}
	else if (_aorb == false) //bot is below plane
	{
		if (((_rel_theta >= 0) && (_rel_theta < M_PI_2))
				|| ((_rel_theta > 3 * M_PI_2) && (_rel_theta < 2 * M_PI)))
		{
#if BOD_DEBUG
			std::cout<<"will apply force"<<std::endl;
#endif
			apply = true;
		}
		else
		{
#if BOD_DEBUG
		std::cout<<"will not apply force"<<std::endl;
#endif
		}
	}
	else
	{
#if BOD_DEBUG
		std::cout<<"broke in applyForce()"<<std::endl; //LMAO
#endif
	}
	return apply;
}
//----------------------------------------------------------------------------------------------------------

// Modifies velocity, location, and resets acceleration with values that
// are given by the three laws.
void Body::update()
{
	//To make the slow down not as abrupt
	// Update velocity
	//velocity.addVector(acceleration);
	// Limit speed
//	velocity.limit(1);
//	velocity.mulScalar(maxSpeed);
	prev_location.set(location.x, location.y);
	ros::Time newTime = ros::Time::now();
	double tstep = newTime.toSec() - curTime.toSec();
	curTime = newTime;
	if (bodyPause)
	{
		bodyPause = false;
		return;
	}

	heading += (b - a) / l * tstep;
	if (heading > 2 * M_PI || heading < 0)
		heading = fmod(heading + 2 * M_PI, 2 * M_PI);
	velocity.set((a + b) / 2 * cos(heading), -(a + b) / 2 * sin(heading));
	Pvector temp(velocity);
	temp.mulScalar(tstep);
	location.addVector(temp);

}

// Run flock() on the flock of bodies.
// This applies the three rules, modifies velocities accordingly, updates data,
// and corrects bodies which are sitting outside of the SFML window
void Body::run(vector<Body> v)
{
	update();
	//elasticCollisions(v);
	inElasticCollisions(v);
	//seperation(v);
	borders();
}

// Checks if bodies go out of the window and if so, wraps them around to
// the other side.
void Body::borders()
{

	//code for hard boundary conditions. Nulls velocity component orthogonal to boundary.
	if ((location.x <= 12) || (location.x >= 288))
	{
		//velocity.x = 0;
		if (location.x <= 12)
		{
			location.x = 12;
		}
		if (location.x >= 288)
		{
			location.x = 288;
		}
	}
	if ((location.y <= 12) || (location.y >= 588))
	{
		//velocity.y = 0;
		if (location.y <= 12)
		{
			location.y = 12;
		}
		if (location.y >= 588)
		{
			location.y = 588;
		}
	}
}

std::pair<float, float> Body::borders(float _fx, float _fy) //applys bounds for the physics engine (adjusts forces)
{
	//code for hard boundary conditions. Nulls velocity component orthogonal to boundary.
	if ((location.x <= 12) || (location.x >= 288))
	{
		_fx = -_fx;
		if (location.x <= 12)
		{
			location.x = 12;
		}
		if (location.x >= 288)
		{
			location.x = 288;
		}
	}
	if ((location.y <= 12) || (location.y >= 588))
	{
		_fy = -_fy;
		if (location.y <= 12)
		{
			location.y = 12;
		}
		if (location.y >= 588)
		{
			location.y = 588;
		}
	}
	std::pair<float, float> that = { _fx, _fy };
	return that;
}

void Body::inElasticCollisions(vector<Body> _bodies) //for collisions between bots, and also the puck.
{
	//Magnitude of separation between bodies
	float desiredseparation = 21;
	for (int i = 0; i < _bodies.size(); i++) // For every body in the system, check if it's too close
	{
		collision = false;
		// Calculate distance from current body to body we're looking at
		float d = location.distance(_bodies.at(i).location);
		float target_sep;
		// If this is a fellow body and it's too close, move away from it
		if ((d <= desiredseparation)
				&& !((id[0] == _bodies.at(i).id[0]) && (id[1] == _bodies.at(i).id[1]))) // (&&d>0))
		{
			if (collision == false)
			{
				collision = true;
			}
#if BOD_DEBUG
			std::cout<<"---BOT-- "<<id[0]<<id[1]<<" With Bot: "<<_bodies.at(i).id[0]<<_bodies.at(i).id[1]<<std::endl;
#endif
			float fnx; //force to be applied (x).
			float fny; //force to be applied (y).

			float dy = _bodies.at(i).location.y - location.y;
			float dx = _bodies.at(i).location.x - location.x;
#if BOD_DEBUG
			std::cout<<"dx,dy: "<<dx<<","<<dy<<std::endl;
#endif
			//angle b/w d and origin.
			float d_angle = angle(Pvector(dx, dy));
			d_angle = angleConvert(d_angle);

			float phi = M_PI_2 + d_angle;
			phi = angleConvert(phi);

#if BOD_DEBUG
			std::cout<<"Angle of seperation: "<<d_angle*180/M_PI<<std::endl;
			std::cout<<"Angle of collision: "<<phi*180/M_PI<<std::endl;
#endif

			int h = quadrant(heading);
			float fx = force * cos(heading); //bot force in heading direction(x).
			float fy = abs(force * sin(heading)); //bot force in heading direction (y).

			if ((h == 1) || (h == 2))
			{
				fy = -fy;
			}
			float mag = sqrt(pow(fx, 2) + pow(fy, 2));
			std::pair<float, float> forces = borders(fx, fy);
			fx = forces.first;
			fy = forces.second;

			float abs_theta = angleConvert(angle(Pvector(fx, fy))); //angle of force in absolute position.
			abs_theta = angleConvert(abs_theta);

			int Q = quadrant(phi); //quadrant bot is in
			bool above_below = aboveOrBelow(dx, dy, Q); //gets quadrant collision is happening.
			float rel_theta = getRelTheta(abs_theta, phi, Q); //velocity direction in frame of the collision plane.
			bool apply_force = applyForce(above_below, rel_theta); //finds if force points towards the plane.

			if (apply_force == true)
			{
				location.x = prev_location.x;
				location.y = prev_location.y;
				fnx = fx - (fx * pow(sin(phi), 2) + fy * sin(phi) * cos(phi));
				fny = fy - (fy * pow(cos(phi), 2) + fx * sin(phi) * cos(phi));
			}
			else
			{
				location.x = prev_location.x;
				location.y = prev_location.y;
				fnx = fx;
				fny = fy;
			}
#if BOD_DEBUG
			std::cout<<"forces direction: "<<abs_theta*180/M_PI<<std::endl;
			std::cout<<"forces (fx,fy) "<<fx<<","<<fy<<std::endl;
			std::cout<<"fn (fnx,fny) "<<fnx<<","<<fny<<std::endl;
			std::cout<<"heading: "<<heading*180/M_PI<<std::endl;
			std::cout<<"relative theta: "<<rel_theta*180/M_PI<<std::endl;
			std::cout<<"sep distance: "<<d<<std::endl;
			std::cout<<"location pre adjustment: "<<location.x<<","<<location.y<<std::endl;
#endif
			location.addVector(Pvector(fnx,fny)); //IMPORTANT
#if BOD_DEBUG
			std::cout<<"adjusted location: "<<location.x<<","<<location.y<<std::endl;
			std::cout<<"----------------------------------------------------"<<std::endl;
#endif
		}
	}
}

float Body::angle(Pvector v)
{
	// From the definition of the dot product. negated to transform to first quadrant from 4th.
	float angle = -1 * (float) (atan2(v.y, v.x));
	return angle;
}
