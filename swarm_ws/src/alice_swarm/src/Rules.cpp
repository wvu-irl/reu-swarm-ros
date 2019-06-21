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
AliceStructs::vel Rules::goToTarget(std::list<AliceStructs::obj> targets, float strength, float fov)
{
	AliceStructs::vel to_return;
	std::pair<float, float> temp_pair1;
	temp_pair1.first = 0;
	temp_pair1.second = 0;
	for (auto &obj : targets)
	{
		//std::cout << "there's shit here" << std::endl;
		//avoids things in direction of travel
		if ((obj.dir < M_PI / 180 * fov || obj.dir > 2 * M_PI - M_PI / 180 * fov) )
		{
			std::pair<float, float> temp_pair2(ROBOT_SIZE * strength / pow(obj.dis, 0.2), obj.dir);
			temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);
		}
	}
	to_return.mag = temp_pair1.first;
	to_return.dir = temp_pair1.second;
	return to_return;
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
AliceStructs::vel Rules::magnetAvoid(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::vel to_return;
	std::pair<float, float> temp_pair1;
	temp_pair1.first = 0;
	temp_pair1.second = 0;
	for (auto &bot : bots)
	{

		std::pair<float, float> temp_pair2(pow(ROBOT_SIZE * strength / (bot.dis - ROBOT_SIZE), 3), M_PI + bot.dir);
		temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);

	}
	to_return.mag = temp_pair1.first;
	to_return.dir = temp_pair1.second;

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
		if ((bot.dir < M_PI / 180 * fov || bot.dir > 2 * M_PI - M_PI / 180 * fov))
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

		//std::cout << "yeeting" << std::endl;
		std::pair<float, float> temp_pair2(pow(ROBOT_SIZE * strength / (obj.dis - ROBOT_SIZE), 3), M_PI + obj.dir);
		temp_pair1 = addPolarVectors(temp_pair1, temp_pair2);

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
