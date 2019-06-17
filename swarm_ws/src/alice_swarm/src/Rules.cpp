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
	float distance = 0;to_return.dir = -1;
	for (auto &bot : bots) {
		direction += bot.dir;
		distance += bot.dis;
	}
	direction = direction/bots.size();
	distance = distance/bots.size();
	if (4 * ROBOT_SIZE < distance * tolerance/SENSE)
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
		if ((bot.dir < M_PI/12 * tolerance/SENSE) || (bot.dir > 2*M_PI - M_PI/12 * tolerance/SENSE)
				&& (bot.dis < ROBOT_SIZE / 2 * tolerance/SENSE))
		{
			to_return.dir = fmod((bot.dir + M_PI),(2 * M_PI));
			to_return.mag = 1 - sin(bot.dir / 2);
			should_ignore = true;
			return to_return;
		}
	}
	to_return.mag = -1;
	return to_return;
}

AliceStructs::vel Rules::avoidObstacles(std::list <AliceStructs::obj> obstacles, float tolerance)
{
	AliceStructs::vel to_return;
	if (obstacles.size() == 0)
	{
		to_return.mag = -1;
		return to_return;
	}
	for (auto &obj : obstacles)
	{
		if ((obj.dir < M_PI/12 * tolerance/SENSE) or (obj.dir > 2 * M_PI - M_PI/12 * tolerance/SENSE)
				and (obj.dis < ROBOT_SIZE / 2 * tolerance))
		{
			to_return.dir = fmod((obj.dir + M_PI),(2 * M_PI));
			to_return.mag = 1 - sin(to_return.dir/2);
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

