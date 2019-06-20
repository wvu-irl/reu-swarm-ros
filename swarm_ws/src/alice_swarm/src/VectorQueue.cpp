#include "alice_swarm/VectorQueue.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>
VectorQueue::VectorQueue(){

}

void VectorQueue::oneToQueue(AliceStructs::ideal toAdd)
{
	vectorQueue.push_back(toAdd);
}

std::pair<float, float> VectorQueue::addPolarVectors(std::pair<float, float> v1, std::pair<float, float> v2)
{
	std::pair<float, float> v;
	float x(v1.first * cos(v1.second) + v2.first * cos(v2.second));
	float y(v1.first * sin(v1.second) + v2.first * sin(v2.second));
	v.first = pow(pow(x, 2) + pow(y, 2), 0.5);
	v.second = atan2(y, x);
	return v;
}

AliceStructs::vel VectorQueue::createCompromise()
{
	AliceStructs::vel to_return;
	float dir;
	float pri;
	//int size = vectorQueue.size();
	//std::cout << "---" << size << std::endl;
	while (!vectorQueue.empty())
	{
		AliceStructs::ideal current = vectorQueue.back();
		//std::cout << current.dir << std::endl;
		vectorQueue.pop_back();
		float current_pri = pow(current.pri / (current.dis/10 + 1), 2);
		dir = (dir * pri + current_pri * current.dir)/(pri + current_pri);
	}
	to_return.dir = dir;
	to_return.mag = 1;
	return to_return;
};
