#ifndef VECTORQUEUE_H_
#define VECTORQUEUE_H_

#include "aliceStructs.h"

class VectorQueue
{
public:

	std::vector<AliceStructs::ideal> vectorQueue; //stores the ideals to be added together

	VectorQueue(); //dummy constructor

	void oneToQueue(AliceStructs::ideal toAdd); //just adds an ideal to the vector

	AliceStructs::vel createCompromise(); //adds the ideals together using a priority based system

private:

	std::pair<float, float> addPolarVectors(std::pair<float, float> v1, std::pair<float, float> v2); //helper for adding vectors together

	AliceStructs::vel past_vector; //deprecated, could be used to keep track of acceleration (change in the velocity commands)

};

#endif /* VECTORQUEUE_H_ */
