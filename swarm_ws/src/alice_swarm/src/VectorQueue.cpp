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

AliceStructs::vel VectorQueue::createCompromise()
{
	float compromise_angle = 0;
	float compromise_speed = 0;
	float priority = 0;

	while (!vectorQueue.empty())
	{
		AliceStructs::ideal current = vectorQueue.back();
		vectorQueue.pop_back();
		float current_priority = pow((current.pri / (current.dis + 1)), 2);
		compromise_angle = (compromise_angle * priority + current.dir * current_priority) / (priority + current_priority);
		compromise_speed = (compromise_speed * priority + current.spd * current_priority) / (priority + current_priority);
		priority += current_priority;
	}
	AliceStructs::vel to_return;
	to_return.dir = (compromise_angle + past_vector.dir)/2;
	to_return.mag = (compromise_speed + past_vector.mag)/2;
	past_vector.dir = to_return.dir;
	past_vector.mag = to_return.mag;
	return to_return;
};
