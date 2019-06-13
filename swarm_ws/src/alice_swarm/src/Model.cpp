#include "Model.h"
#include "aliceStructs.h"
#include <ros/ros.h>

class Model
{

ideal Model::generateIdeal()
{
	ideal toReturn;
	toReturn.dir = -1;
	int tolerance = 1;
	while ((toReturn.dir == -1) && ros::ok)
	{
		std::list <vel> ideal_list = {rules.avoidObstacles(obstacles, tolerance),
			rules.avoidRobots(robots, tolerance),
			rules.maintainSpacing(robots, tolerance)};
		for (int i = 0; i < ideal_list.size(); i++)
		{
			if (ideal_list.get(i).dis != -1)
			{
				to_return.dir = ideal_list.get(i).dir;
				to_return.spd = ideal_list.get(i).mag;
				to_return.dis = 0;
				