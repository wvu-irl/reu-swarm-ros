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
	std::pair<float, float> compromise;
	compromise.first =0;
	compromise.second=0;
	int size = vectorQueue.size();
	//std::cout << "---" << size << std::endl;
	while (!vectorQueue.empty())
	{
		std::pair<float, float> temp;
		AliceStructs::ideal current = vectorQueue.back();
		vectorQueue.pop_back();
		temp.first= pow((current.spd / (current.dis + 1)), 3);
		temp.second = current.dir;
		//std:: cout << temp.first << " " << temp.second << std::endl;
		if (temp.first > 0.01) compromise = addPolarVectors(compromise, temp);
	}

	to_return.dir = compromise.second;
	to_return.mag = compromise.first;
	return to_return;
};
