#include "alice_swarm/Rules.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

Rules::Rules()
{
	should_ignore = false;
}
AliceStructs::vel Rules::dummy1()
{
	AliceStructs::vel to_return;
	to_return.mag = 1;
	to_return.dir = 0;
	return to_return;
}
std::pair<float, float> Rules::addPolarVectors(std::pair<float, float> v1, std::pair<float, float> v2)
{
	std::pair<float, float> v;
	float x(v1.first * cos(v1.second) + v2.first * cos(v2.second));
	float y(v1.first * sin(v1.second) + v2.first * sin(v2.second));
	v.first = pow(pow(x, 2) + pow(y, 2), 0.5);
	v.second = atan2(y, x);
	return v;
}

AliceStructs::vel Rules::maintainSpacing(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::vel to_return;
	std::pair<float, float> temp_pair1;
	temp_pair1.first = 0;
	temp_pair1.second = 0;
	for (auto &bot : bots)
	{
		std::pair<float, float> temp_pair2(bot.dis, bot.dir);

		temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);
	}
	to_return.mag = temp_pair1.first / bots.size() * strength;

	to_return.dir = temp_pair1.second;
	return to_return;
}

AliceStructs::vel Rules::magnetAvoid(std::list<AliceStructs::neighbor> bots, float strength, float fov)
{
	AliceStructs::vel to_return;
	std::pair<float, float> temp_pair1;
	temp_pair1.first = 0;
	temp_pair1.second = 0;
	for (auto &bot : bots)
	{
		if (bot.dis < ROBOT_SIZE + strength)
		{
			std::pair<float, float> temp_pair2(pow(ROBOT_SIZE * strength / bot.dis, 3), M_PI + bot.dir);
			temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);
		}
	}
	to_return.mag = temp_pair1.first;
	to_return.dir = temp_pair1.second;
	std::cout << to_return.mag << " - " << to_return.dir << std::endl;
	return to_return;
}

AliceStructs::vel Rules::birdAvoid(std::list<AliceStructs::neighbor> bots, float strength, float fov)
{
	AliceStructs::vel to_return;
	std::pair<float, float> temp_pair1;
	temp_pair1.first = 0;
	temp_pair1.second = 0;
	for (auto &bot : bots)
	{
		//avoids things in direction of travel
		if ((bot.dir <M_PI/180*fov || bot.dir > 2 * M_PI - M_PI / 180 * fov) && (bot.dis < 2 * (ROBOT_SIZE + strength)))
		{
			std::pair<float, float> temp_pair2(pow(ROBOT_SIZE * strength / bot.dis, 3), M_PI + bot.dir);
			temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);
		}
	}
	to_return.mag = temp_pair1.first;
	to_return.dir = temp_pair1.second;
	return to_return;
}
//
AliceStructs::vel Rules::avoidObstacles(std::list<AliceStructs::obj> obstacles, float strength, float fov)
{
	AliceStructs::vel to_return;
		std::pair<float, float> temp_pair1;
		temp_pair1.first = 0;
		temp_pair1.second = 0;
		for (auto &obj : obstacles)
		{
			//std::cout << "there's shit here" << std::endl;
			//avoids things in direction of travel
			if ((obj.dir <M_PI/180*fov || obj.dir > 2 * M_PI - M_PI / 180 * fov) && (obj.dis < ROBOT_SIZE * strength))
			{
				std::pair<float, float> temp_pair2(pow(ROBOT_SIZE * strength / obj.dis, 3), M_PI + obj.dir);
				temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);
			} else if (obj.dis < ROBOT_SIZE * 1.5)
			{
				//std::cout << "yeeting" << std::endl;
				std::pair<float, float> temp_pair2(pow(ROBOT_SIZE * strength / obj.dis, 3), M_PI + obj.dir);
				temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);
			}
		}
		to_return.mag = temp_pair1.first;
		to_return.dir = temp_pair1.second;
		return to_return;
}

/*
 vel goToTarget(std::list <obs> targets, float tolerance)
 {
 vel to_return;

 }*/
