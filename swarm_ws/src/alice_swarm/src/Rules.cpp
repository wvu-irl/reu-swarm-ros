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
std::pair<float,float> Rules::addPolarVectors(std::pair<float,float> v1, std::pair<float,float> v2)
{
	std::pair<float,float> v;
	float x(v1.first * cos(v1.second) + v2.first * cos(v2.second));
	float y(v1.first * sin(v1.second) + v2.first * sin(v2.second));
	v.first= pow(pow(x, 2) + pow(y, 2), 0.5);
	v.second = atan2(y, x);
}

AliceStructs::vel Rules::maintainSpacing(std::list<AliceStructs::neighbor> bots, float strength)
{
	AliceStructs::vel to_return;
	std::pair<float,float> temp_pair1;
	temp_pair1.first=0;
	temp_pair1.second=0;
	for (auto &bot : bots)
	{
		std::pair<float, float> temp_pair2(bot.dis,bot.dir);
		temp_pair1=addPolarVectors(temp_pair1,temp_pair2);
	}
	to_return.mag=temp_pair1.first/bots.size()*strength;
	to_return.dir=temp_pair1.second;
	return to_return;
}
//
//AliceStructs::vel Rules::avoidNeighbors(std::list<AliceStructs::neighbor> bots, float tolerance)
//{
//	AliceStructs::vel to_return;
//	for (auto &bot : bots)
//	{
//		if ((bot.dir < M_PI / 12 * tolerance)
//				|| (bot.dir > 2 * M_PI - M_PI / 12 * tolerance) && (bot.dis < ROBOT_SIZE / 2 * tolerance )
//				{
//					to_return.dir = fmod((bot.dir + M_PI), (2 * M_PI));
//					to_return.mag = 1 - sin(bot.dir / 2);
//					should_ignore = true;
//					return to_return;
//				}
//			}
//			to_return.mag = 0;
//			to_return.dir = 0;
//			return to_return;
//		}
//	}
//}
//
//AliceStructs::vel Rules::avoidObstacles(std::list<AliceStructs::obj> obstacles, float tolerance)
//{
//	AliceStructs::vel to_return;
//	for (auto &obj : obstacles)
//	{
//		if ((obj.dir < M_PI / 12 * tolerance)
//				or (obj.dir > 2 * M_PI - M_PI / 12 * tolerance) and (obj.dis < ROBOT_SIZE / 2 * tolerance))
//		{
//			to_return.dir = fmod((obj.dir + M_PI), (2 * M_PI));
//			to_return.mag = 1 - sin(to_return.dir / 2);
//			return to_return;
//		}
//	}
//	to_return.mag = 0;
//	to_return.dir = 0;
//	return to_return;
//}

/*
 vel goToTarget(std::list <obs> targets, float tolerance)
 {
 vel to_return;

 }*/

