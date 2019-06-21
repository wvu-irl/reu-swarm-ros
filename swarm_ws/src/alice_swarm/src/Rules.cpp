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
	float x = (i1.spd * cos(i1.dir) * i1.pri + i2.spd * cos(i2.dir) * i2.pri)/(i1.pri + i2.pri);
	float y = (i1.spd * sin(i1.dir) * i1.pri + i2.spd * sin(i2.dir) * i2.pri)/(i1.pri + i2.pri);
	to_return.spd = pow(pow(x, 2) + pow(y, 2), 0.5);
	to_return.dir = atan2(y, x);
	to_return.pri = i1.pri + i2.pri;
	std::cout << to_return.dir << std::endl;
	return to_return;
}

AliceStructs::ideal Rules::maintainSpacing(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	to_return.spd = 1;
	for (auto& bot : bots)
	{
		to_return.dir += bot.dir/bots.size();
		to_return.pri += bot.dis/bots.size();
	}
	to_return.pri = pow(to_return.pri * strength, 0.5);
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
		if (bot.dir < ROBOT_SIZE*3) {
			to_return.spd -= 0.2;
		}
		temp_pri = 10 * strength / pow(2, bot.dis - 3/2*ROBOT_SIZE);
		to_return.dir = (to_return.dir * to_return.pri + temp_pri * bot.dir)/
				(temp_pri + to_return.pri);
		to_return.pri += temp_pri;
	}
	to_return.dir = fmod(to_return.dir + M_PI, 2*M_PI);
	to_return.spd = abs(to_return.spd);
	return to_return;
}

AliceStructs::ideal Rules::birdAvoid(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::ideal to_return;

	float pri = 0;
	float temp_pri;
	for (auto &bot : bots)
	{
		temp_pri = strength/((bot.dis - ROBOT_SIZE/2) * abs(bot.dir - M_PI) + 1);
		if (temp_pri > pri)
		{
			pri = temp_pri;
			to_return.dir = fmod(bot.dir + M_PI, 2*M_PI);
			to_return.spd = 1/temp_pri;
		}
	}
	to_return.pri = temp_pri;
	return to_return;
}

AliceStructs::ideal Rules::goToTarget(std::list<AliceStructs::obj> targets, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0.0001;
	to_return.dir = 0.0001;
	for (auto& tar : targets)
	{
		float temp_pri = strength * pow(tar.dis, 0.5);
		to_return.dir = (to_return.dir * to_return.pri + tar.dir * temp_pri)/(to_return.pri + temp_pri);
		to_return.pri += temp_pri;
	}
	return to_return;
}

AliceStructs::ideal Rules::followFlow(std::list<AliceStructs::ideal> flows, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0;
	to_return.dir = 0;
	for (auto& flow : flows)
	{
		float temp_pri = flow.spd*strength/(flow.dis+10);
		to_return.dir = (to_return.dir * to_return.pri + flow.dir * temp_pri)/(to_return.pri + temp_pri);
		to_return.spd = (to_return.spd * to_return.pri + flow.spd * temp_pri)/(to_return.pri + temp_pri);
		to_return.pri += temp_pri;
	}

	return to_return;
}


AliceStructs::ideal Rules::avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength)
{
	AliceStructs::ideal to_return;
	to_return.pri = 0.0001;
	to_return.dir = 0.0001;
	to_return.spd = 1;
	//to_return.spd = 0;
	for (auto &obs : obstacles)
	{
		if (obs.dis < 4 * ROBOT_SIZE)
		{
			to_return.spd = 1 / to_return.pri;
			to_return.dir = fmod(obs.dir + M_PI, 2*M_PI);
			to_return.pri = 10 * strength / pow(2, obs.dis - 3 * ROBOT_SIZE / 2);
		}
	}
	return to_return;
}

/*
 ideal goToTarget(std::list <obs> targets, float tolerance)
 {
 ideal to_return;

 }*/
