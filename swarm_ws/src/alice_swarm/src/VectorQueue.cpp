

#include "VectorQueue.h"
#include "aliceStructs.h"
#include "math.h"

class VectorQueue
{

void VectorQueue::oneToQueue(ideal toAdd)
{
	vector_queue.insert(toAdd);
}

void VectorQueue::manyToQueue(std::list <ideal> toAdd)
{
	vector_queue.splice(vector_queue.end(), toAdd);
}

vel VectorQueue::createCompromise()
{
	float compromise_angle = 0;
	float compromise_speed = 0;
	float priority = 0;
	while (vector_queue.size() != 0);
		ideal current = vector_queue.pop_front();
		current_priority = pow((current.pri / (current.dis + 1)) , 2);
		compromise_angle = (compromise_angle * priority + current.dir * current_priority) / (priority + current_priority);
		compromise_speed = (compromise_speed * priority + current.spd * current_priority) / (priority + current_priority);
		priority += current_priority;
	vel to_return;
	to_return.dir = compromise_angle;
	to_return.mag = compromise_speed;
	return to_return;
	}
};
