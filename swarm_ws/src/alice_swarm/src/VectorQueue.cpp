#include "alice_swarm/VectorQueue.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>
VectorQueue::VectorQueue()
{
}

void VectorQueue::oneToQueue(AliceStructs::ideal toAdd)
{
	vectorQueue.push_back(toAdd);
}

std::pair<float, float> VectorQueue::addPolarVectors(std::pair<float, float> v1, std::pair<float, float> v2)
{
	std::pair<float, float> v;
	float x(v1.first * cos(v1.second) + v2.first * cos(v2.second)); //converts to x y and adds
	float y(v1.first * sin(v1.second) + v2.first * sin(v2.second));
	v.first = pow(pow(x, 2) + pow(y, 2), 0.5); //converts them back
	v.second = atan2(y, x);
	return v;
}

AliceStructs::vel VectorQueue::createCompromise()
{
	AliceStructs::vel to_return;
	float dir;
	float pri;
	float spd;
	//int size = vectorQueue.size();
	//std::cout << "---" << size << std::endl;
	while (!vectorQueue.empty())
	{
		AliceStructs::ideal current = vectorQueue.back();
		//std::cout << current.dir << std::endl;
		vectorQueue.pop_back();

		//math to figure out how important the vector to be followed is
		float current_pri = current.pri / (1+pow(current.dis / 10,2));
		dir = (dir * pri + current_pri * current.dir) / (pri + current_pri);
		spd = (spd * pri + current_pri * current.spd) / (pri + current_pri);
	}
	to_return.dir = dir;
	to_return.mag = spd;

	return to_return;
}
;
