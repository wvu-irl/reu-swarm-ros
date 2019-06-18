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

private:

	std::pair<float, float> addPolarVectors(std::pair<float, float> v1, std::pair<float, float> v2);


	AliceStructs::vel past_vector;
};

#endif /* VECTORQUEUE_H_ */
