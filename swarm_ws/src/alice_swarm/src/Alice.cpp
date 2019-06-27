#include "alice_swarm/Alice.h"
#include "alice_swarm/Rules.h"
#include "alice_swarm/Model.h"
#include <iostream>

Alice::Alice(AliceStructs::mail _data)
{
	name = _data.name;
	model = Model(_data.name);
	rules = Rules(model);
	model.updateModel(_data);
}

void Alice::updateModel(AliceStructs::mail _data)
{
	model.updateModel(_data);
}

AliceStructs::vel generateVel()
{
	rules.generateVel();
}
