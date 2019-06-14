#ifndef MODEL_H_
#define MODEL_H_

#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Rules.h"

class Model
{
public:
	std::list <AliceStructs::obj> obstacles;
	std::list <AliceStructs::neighbor> robots;
	std::list <AliceStructs::obj> targets;

	Model();//dummy

	Model(int _name);

	AliceStructs::ideal generateIdeal();

	void addToModel(AliceStructs::mail toAdd);

	void clear();

private:
	Rules rules;
	int name;
};

#endif /* MODEL_H_ */
