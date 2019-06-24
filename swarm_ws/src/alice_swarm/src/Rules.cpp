#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

Rules::Rules()
{
	should_ignore = false;
}
AliceStructs::ideal Rules::dummy1()
{
	AliceStructs::ideal to_return;
	to_return.spd = 1;
	to_return.dir = 0;
	return to_return;
}
AliceStructs::ideal Rules::addIdeals(AliceStructs::ideal i1, AliceStructs::ideal i2)
{
	AliceStructs::ideal to_return;
	float x = (i1.spd * cos(i1.dir) * i1.pri + i2.spd * cos(i2.dir) * i2.pri) / (i1.pri + i2.pri);
	float y = (i1.spd * sin(i1.dir) * i1.pri + i2.spd * sin(i2.dir) * i2.pri) / (i1.pri + i2.pri);
	to_return.spd = pow(pow(x, 2) + pow(y, 2), 0.5);
	to_return.dir = atan2(y, x);
	to_return.pri = i1.pri + i2.pri;
	//std::cout << to_return.dir << std::endl;
	return to_return;
}

AliceStructs::ideal Rules::maintainSpacing(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 1;
	const float BRAKING = 3; //due to latency issues, we need to slow this rule down. Increase this number
	//if the robots are hitting each other sometimes, or decrease it to slow things down.
	for (auto& bot : bots)
	{
		to_return.dir += bot.dir / bots.size();
		to_return.pri += bot.dis / bots.size();
		to_return.spd += (pow(strength, BRAKING) / (0 - pow(bot.dis + pow(strength, BRAKING / 2) - ROBOT_SIZE, 2)) + 1)
		    / bots.size();
	}
	to_return.spd -= 1;
	to_return.pri = pow(to_return.pri * strength - ROBOT_SIZE / 2, 0.2) / 3;
	return to_return;
}

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
		temp.pri = strength / pow(2, bot.dis - 3 / 2 * ROBOT_SIZE);
		temp.spd = pow(strength, 2) / (0 - pow(bot.dis + strength - ROBOT_SIZE, 2)) + 1;
		temp.dir = bot.dir;
		to_return = addIdeals(temp, to_return);
		//to_return.pri += temp_pri;
		//std::cout << to_return.dir << " - " << bot.dir << std::endl;
	}
	//to_return.pri = to_return.pri;
	to_return.dir = fmod(to_return.dir + M_PI, 2 * M_PI);
	//to_return.spd = abs(to_return.spd);
	return to_return;
}

AliceStructs::ideal Rules::birdAvoid(std::list<AliceStructs::neighbor> bots, float strength)
{ //I don't know how or why this method works. Fair warning.
	AliceStructs::ideal to_return;

	//float pri = 16;
	to_return.pri = strength / pow(2, bots.front().dis - 3 / 2 * ROBOT_SIZE);
	to_return.dir = fmod(bots.front().dir + M_PI, 2 * M_PI);
	to_return.spd = pow(strength, 2)
	    / (0 - (bots.front().dis + strength - ROBOT_SIZE) * (abs(bots.front().dir + strength - ROBOT_SIZE))) + 1;
	/*for (auto &bot : bots)
	 {
	 temp_pri = strength/((bot.dis - ROBOT_SIZE/2) * (abs(bot.dir - M_PI)/2) + 1);
	 if (temp_pri > pri)
	 {
	 std::cout << bot.dis << " - " << bot.dir << std::endl;
	 pri = temp_pri;
	 to_return.dir = fmod(bot.dir + M_PI, 2*M_PI);
	 to_return.spd = pow(strength, 2) / (0 - (bot.dis + strength - ROBOT_SIZE) *
	 (abs(bot.dir + strength - ROBOT_SIZE))) + 1;
	 }
	 }*/
	//to_return.pri = temp_pri;
	return to_return;
}

AliceStructs::ideal Rules::goToTarget(std::list<AliceStructs::obj> targets, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0.001;
	to_return.dir = 0.001;
	to_return.spd = 0.001;
	if (targets.size() != 0)
	{
		for (auto& tar : targets)
		{
			if (tar.dis > to_return.pri)
				to_return.dir = tar.dir;
			to_return.pri = tar.dis;
			to_return.spd = (pow(strength, 2) / (0 - pow(tar.dis + strength, 2)) + 1);
		}
	to_return.spd -= 1;
	to_return.pri = pow(to_return.pri * strength, 0.2) / 3;
	}
return to_return;
}

AliceStructs::ideal Rules::followFlow(std::list<AliceStructs::ideal> flows, float strength)
{
AliceStructs::ideal to_return;
to_return.pri = 0.001;
to_return.dir = 0.001;
to_return.spd = 0.001;
if (flows.size() != 0)
{
	for (auto &flow : flows)
	{
		float temp_pri = flow.spd * strength / (flow.dis + 10);
		to_return.dir = (to_return.dir * to_return.pri + flow.dir * temp_pri) / (to_return.pri + temp_pri);
		to_return.spd = (to_return.spd * to_return.pri + flow.spd * temp_pri) / (to_return.pri + temp_pri);
		to_return.pri += temp_pri;
	}
}
return to_return;
}

AliceStructs::ideal Rules::avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength)
{
AliceStructs::ideal to_return;
to_return.pri = 0.001;
to_return.dir = 0.001;
to_return.spd = 1;
for (auto &obs : obstacles)
{
	if (obs.dis < 2 * ROBOT_SIZE)
	{
		AliceStructs::ideal temp;
		temp.pri = strength / pow(2, obs.dis - 3 / 2 * ROBOT_SIZE);
		temp.spd = pow(strength, 2) / (0 - pow(obs.dis + strength - ROBOT_SIZE, 2)) + 1;
		temp.dir = obs.dir;
		to_return = addIdeals(temp, to_return);
	}
}
to_return.dir = fmod(to_return.dir + M_PI, 2 * M_PI);
return to_return;
}

/*
 ideal goToTarget(std::list <obs> targets, float tolerance)
 {
 ideal to_return;

 }*/
