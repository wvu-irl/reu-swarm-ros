
#ifndef VECTORQUEUE_H_
#define VECTORQUEUE_H_

#include "aliceStructs.h"

class VectorQueue
{
	public:

		std::list <ideal> vector_queue;

		VectorQueue();

		void addToQueue(ideal toAdd);

		vel createCompromise();
	};

#endif /* VECTORQUEUE_H_ */
