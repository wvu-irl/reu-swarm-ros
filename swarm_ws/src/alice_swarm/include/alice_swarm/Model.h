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
	std::list <AliceStructs::ideal> flows;
	Rules rules;

	Model();//dummy


	Model(int _name);

	AliceStructs::ideal generateIdeal();

	void addToModel(AliceStructs::mail toAdd);

	void clear();

private:
	int name;
	void addPolarVel(AliceStructs::vel &_vel1, AliceStructs::vel &_vel2);
	void normalize(AliceStructs::vel &_vel);
	void dDriveAdjust(AliceStructs::vel &_vel); //assumes that vector has already been normalized
};

#endif /* MODEL_H_ */
