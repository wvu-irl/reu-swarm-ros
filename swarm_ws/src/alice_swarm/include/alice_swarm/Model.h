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

	Model();//dummy


	Model(int _name,int _sid);

	AliceStructs::ideal generateIdeal();

	void addToModel(AliceStructs::mail toAdd);

	void clear();
	Rules rules;
private:

	int name;
	void addIdeal(AliceStructs::ideal &_ideal1, AliceStructs::ideal &_ideal2);
	int sid;

	void normalize(AliceStructs::vel &_vel);
	void dDriveAdjust(AliceStructs::vel &_vel); //assumes that vector has already been normalized
};

#endif /* MODEL_H_ */
