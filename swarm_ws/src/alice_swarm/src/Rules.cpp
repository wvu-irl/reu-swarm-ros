#include "Rules.h"
#include "aliceStructs.h"
#include "math"

vel Rules::maintainSpacing(std::list <neighbor> bots, float tolerance)
{
	vel to_return;
	if (bots.size() == 0)
	{
		to_return.dis = -1;
		return to_return;
	}
	float direction = 0;
	float distance = 0;to_return.dis = -1;
	for (auto &bot : bots) {
		direction += bot.dir;
		distance += bot.dis;
	}
	direction = direction/bots.size();
	distance = distance/bots.size();
	if (4 * ROBOT_SIZE / tolerance < distance * SENS)
	{
		to_return.dir = direction;
		to_return.dis = distance;
		return to_return;
	}
	else
	{
		to_return.dis = -1;
		return to_return;
	}
}

vel avoidRobots(std::list <neighbor> bots, float tolerance)
{

}


