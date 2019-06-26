/*
 * VectorQueue
 *
 * This class stores ideal vectors and uses them to generate a velocity vector to send to
 * the low-level controllers
 *
 * Authors: Casey Edmonds-Estes and Jeongwoo Seo
 */

#ifndef VECTORQUEUE_H_
#define VECTORQUEUE_H_

#include "aliceStructs.h"

class VectorQueue
{
public:

	std::vector<AliceStructs::ideal> vectorQueue; //Stores the ideal vectors

	VectorQueue();

	/*
	 * Adds a single ideal vector to the queue.
	 */
	void oneToQueue(AliceStructs::ideal toAdd);

	/*
	 * This method takes a weighted average of the ideal vectors in the queue, which
	 * it then uses to generate a velocity vector
	 */
	AliceStructs::vel createCompromise();
};

#endif /* VECTORQUEUE_H_ */
