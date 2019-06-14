#ifndef VECTORQUEUE_H_
#define VECTORQUEUE_H_

#include "aliceStructs.h"

class VectorQueue
{
public:

	std::vector<AliceStructs::ideal> vectorQueue;

	VectorQueue();

	void oneToQueue(AliceStructs::ideal toAdd);

	AliceStructs::vel createCompromise();
};

#endif /* VECTORQUEUE_H_ */
