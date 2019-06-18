#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

Rules::Rules(){
	should_ignore = false;
}

AliceStructs::vel Rules::maintainSpacing(std::list <AliceStructs::neighbor> bots, float tolerance)
{
	AliceStructs::vel to_return;
	if (bots.size() == 0)
	{
		to_return.mag = -1;
		return to_return;
	}
	float direction = 0;
	float distance = 0;
	float x = 0;
	float y = 0;
	for (auto &bot : bots) {
		x += bot.dis * cos(bot.dir);
		y += bot.dis * sin(bot.dir);
	}
	x = x/bots.size();
	y = y/bots.size();
	distance = pow((pow(x, 2) + pow(y, 2)), 0.5);
	if (distance < 1 && tolerance > 100)
	{
		to_return.dir = 0;
		to_return.mag = 0;
		return to_return;
	}
	direction = atan2(y, x);
	if (SENSE * ROBOT_SIZE / 5 * tolerance < distance)
	{
		to_return.dir = direction;
		to_return.mag = 1 - sin(direction/2);
		return to_return;
	}
	else
	{
		to_return.mag = -1;
		return to_return;
	}
}

AliceStructs::vel Rules::predictiveAvoid(std::list <AliceStructs::neighbor> bots, float tolerance)
{
	AliceStructs::vel to_return;
	if (bots.size() == 0)
	{
		to_return.mag = -1;
		return to_return;
	}
	for (auto &bot : bots)
	{
		if ((bot.dir < M_PI/6 * tolerance/SENSE) || (bot.dir > 6*M_PI - M_PI/12 * tolerance/SENSE)
				&& (bot.dis < ROBOT_SIZE * SENSE / tolerance))
		{
			to_return.dir = fmod((bot.dir + M_PI), (2 * M_PI));
			to_return.mag = sin(bot.dir / 2);
			should_ignore = true;
			return to_return;
		}
	}
	to_return.mag = -1;
	return to_return;
}

AliceStructs::vel Rules::avoidObstacles(std::list <AliceStructs::obj> obstacles, float tolerance)
{
	//std::cout << "i am supposed to see something soon" << std::endl;
	AliceStructs::vel to_return;
	if (obstacles.size() == 0)
	{
		to_return.mag = -1;
		return to_return;
	}
	for (auto &obj : obstacles)
	{
		if ((obj.dir < M_PI/12 * tolerance/SENSE) or (obj.dir > 2 * M_PI - M_PI/12 * tolerance/SENSE)
				and (obj.dis < ROBOT_SIZE / SENSE * tolerance * 2))
		{
			to_return.dir = fmod((obj.dir + M_PI),(2 * M_PI));
			to_return.mag = sin(obj.dir/2);
			return to_return;
		}
	}
	to_return.mag = -1;
	return to_return;
};

AliceStructs::vel Rules::panicAvoid(std::list <AliceStructs::neighbor> bots, float tolerance)
{
	AliceStructs::vel to_return;
	if (bots.size() == 0)
	{
		to_return.mag = -1;
		return to_return;
	}
	for (auto &bot : bots)
	{
		if (bot.dis < ROBOT_SIZE + 3 + tolerance/SENSE)
		{
			to_return.dir = fmod((bot.dir + M_PI),(2 * M_PI));
			//std::cout << bot.dir << " - " << to_return.dir << std::endl;
			to_return.mag = sin(bot.dir/2);
			should_ignore = true;
			return to_return;
		}
	}
	to_return.mag = -1;
	return to_return;
}

/*
vel goToTarget(std::list <obs> targets, float tolerance)
{
	vel to_return;

}*/

