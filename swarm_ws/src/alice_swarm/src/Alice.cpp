#include "alice_swarm/Alice.h"
#include "alice_swarm/Rules.h"
#include "alice_swarm/Model.h"
#include "wvu_swarm_std_msgs"
#include <iostream>

Alice::Alice(AliceStructs::mail _data)
{
	name = _data.name;
	model = Model(_data.name);
	rules = Rules(model);
	model.updateModel(_data);
}

void Alice::updateModel(wvu_swarm_std_msgs::mail _data)
{
	model.updateModel(_data);
}

AliceStructs::vel generateVel()
{
	rules.generateVel();
}
