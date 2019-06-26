#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

/*
 * Dummy constructor for compiler
 */
Rules::Rules()
{
	should_ignore = false;
}

Rules::Rules(int _sid)
{
	should_ignore = false;
	sid = _sid;
}

/*
 * Dummy rule for testing other parts of Alice
 */
AliceStructs::ideal Rules::dummy1()
{
	AliceStructs::ideal to_return;
	to_return.spd = 1;
	to_return.dir = 0;
	return to_return;
}

/*
 * Performs weighted vector additon to add two ideal vectors together, then returns the result.
 */
AliceStructs::ideal Rules::addIdeals(AliceStructs::ideal i1, AliceStructs::ideal i2)
{
	AliceStructs::ideal to_return;
	float x = (i1.spd * cos(i1.dir) * i1.pri + i2.spd * cos(i2.dir) * i2.pri) / (i1.pri + i2.pri);
	float y = (i1.spd * sin(i1.dir) * i1.pri + i2.spd * sin(i2.dir) * i2.pri) / (i1.pri + i2.pri);
	to_return.spd = pow(pow(x, 2) + pow(y, 2), 0.5);
	to_return.dir = atan2(y, x);
	to_return.pri = i1.pri + i2.pri;
	return to_return;
}


/*
 * Makes bots converge to the center of the swarm. Usually counterbalanced with magnetAvoid, which does the opposite.
 */

AliceStructs::ideal Rules::maintainSpacing(std::list<AliceStructs::neighbor> bots, float strength, float distance)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 1;
	float x = 0;
	float y = 0;
	float pri = 0;
	const float BRAKING = 3; //due to latency issues, we often need to slow this rule down. Increase this number
	//if the robots are hitting each other sometimes, or decrease it to slow things down.
	float count =0;
	for (auto& bot : bots)
	{
		if (bot.sid == sid)
		{
			x += bot.dis * cos(bot.dir);
			y += bot.dis * sin(bot.dir);
			count++;
		}
	}
	//This garbage fire of an equation produces a curve that looks like a translated square root curve capped at 1.
	if (count ==0) return to_return;
	x /= count;
	y /= count;
	float d = pow(pow(x,2)+pow(y,2),0.5);
	to_return.spd = 1;//(pow(strength, BRAKING) / (0 - pow(pow(pow(x, 2) + pow(y, 2), 0.5)
//			+ pow(strength, BRAKING / 2) - ROBOT_SIZE, 2)) + 1)
//	    / bots.size();
	if (d >distance) to_return.dir = fmod(atan2(y, x) + 2*M_PI, 2 * M_PI);
	else to_return.dir = fmod(atan2(y, x) + M_PI, 2 * M_PI);
	to_return.pri = pow(abs(d-distance)* strength, 0.2)/1000;
	return to_return;
}

/*
 * Uses an electromagnetic-style "force" to prevents robots from getting too close.
 */
AliceStructs::ideal Rules::magnetAvoid(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 1;
	float temp_pri;
	for (auto &bot : bots)
	{
		AliceStructs::ideal temp;
		temp.pri = strength / pow(bot.dis - ROBOT_SIZE,2);
		temp.spd = 1;//pow(strength, 2) / (0 - pow(bot.dis + strength - ROBOT_SIZE, 2)) + 1;
		temp.dir = bot.dir;
		to_return = addIdeals(temp, to_return);
	}
	to_return.dir = fmod(to_return.dir + M_PI, 2 * M_PI);
	if (to_return.pri < 0.001) // Handles the possibility that a robots has no neighbors
	{
		to_return.pri = 0;
		to_return.dir = 0;
		to_return.spd = 0;
	}
	return to_return;
}

AliceStructs::ideal Rules::birdAvoid(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::ideal to_return;
	to_return.dir = 0.001;
	to_return.pri = 0.001;
	to_return.spd = 1;
	for (auto &bot : bots)
	{
		float temp_pri = pow(strength, 2) / ((bot.dis-3/2 * ROBOT_SIZE) * ((abs(bot.dir - M_PI) + 1)));
		if (temp_pri > to_return.pri)
		{
			to_return.pri = temp_pri;
			to_return.spd = pow(strength, 2) / (0 - ((bot.dis + strength - ROBOT_SIZE) * (abs(bot.dir - M_PI) + 1))) + 1;
			to_return.dir = fmod(bot.dir + M_PI, 2);
		}
	}
	return to_return;
}

/*
 * Moves a robot towards a target
 */
AliceStructs::ideal Rules::goToTarget(std::list<AliceStructs::obj> targets, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 0;
	float temp = 60;
	if (targets.size() != 0)
	{
		for (auto& tar : targets)
		{
			if (tar.dis < temp)
				to_return.dir = tar.dir;
			// Produces another capped inverse polynomial
			to_return.spd = (pow(strength, 2) / (0 - pow(tar.dis + strength - ROBOT_SIZE/2, 2))) + 1;
			to_return.pri = strength*atan(tar.dis / ROBOT_SIZE/3) * M_PI_2;
		}

	}
	return to_return;
}

/*
 * Makes the robots follow a flow, or a vector with a position
 */
AliceStructs::ideal Rules::followFlow(std::list<AliceStructs::ideal> flows, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 0;
	if (flows.size() != 0)
	{
		for (auto &flow : flows)
		{
			AliceStructs::ideal temp;
			temp.dir = flow.dir;
			temp.spd = flow.spd;
			temp.pri = strength * flow.pri / (pow(flow.dis, 2) + ROBOT_SIZE);

			to_return = addIdeals(to_return, temp);
		}
	}
	return to_return;
}

/*
 * Makes the robot avoid obstacles if they get too close.
 */
AliceStructs::ideal Rules::avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 1;
	for (auto &obs : obstacles)
	{
		if (obs.dis < 2 * ROBOT_SIZE)
		{
			AliceStructs::ideal temp;
			temp.pri = strength / pow(2, obs.dis - 3 / 2 * ROBOT_SIZE);
			temp.spd = 1;
			temp.dir = obs.dir;
			to_return = addIdeals(temp, to_return);
		}
	}
	to_return.dir = fmod(to_return.dir + M_PI, 2 * M_PI);
	return to_return;
}
