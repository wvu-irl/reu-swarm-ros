#include "alice_swarm/VectorQueue.h"
#include "alice_swarm/aliceStructs.h"
#include <math.h>
#include <iostream>

VectorQueue::VectorQueue()
{
}

/*
 * Adds a single ideal vector to the queue.
 */
void VectorQueue::oneToQueue(AliceStructs::ideal toAdd)
{
	vectorQueue.push_back(toAdd);
}

/*
 * This method takes a weighted average of the ideal vectors in the queue, which
 * it then uses to generate a velocity vector
 */
AliceStructs::vel VectorQueue::createCompromise()
{
	AliceStructs::vel to_return;
	float dir;
	float pri;
	float spd;
	while (!vectorQueue.empty())
	{
		AliceStructs::ideal current = vectorQueue.back();
		vectorQueue.pop_back();
		// Factors distance into priority, since vectors from nearby robots are more important.
		float current_pri = current.pri / (1 + pow(current.dis / 10, 2));
		dir = (dir * pri + current_pri * current.dir) / (pri + current_pri);
		spd = (spd * pri + current_pri * current.spd) / (pri + current_pri);
	}
	to_return.dir = dir;
	to_return.mag = spd;
	return to_return;
}
