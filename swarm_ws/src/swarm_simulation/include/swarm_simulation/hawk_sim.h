#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <wvu_swarm_std_msgs/chargers.h>
#include <wvu_swarm_std_msgs/priorities.h>


class Hawk_Sim
{
private:
	bool first = true;
	//palce holders for subscription data ---------------
	wvu_swarm_std_msgs::chargers temp_chargers;
	wvu_swarm_std_msgs::priorities temp_priorities;
	//---------------------------------------------------

	//callback functions
	void chargersCallback(const wvu_swarm_std_msgs::chargers &msg);
	void priorityCallback(const wvu_swarm_std_msgs::priorities &msg);

	//initializer functions
	void makeChargers(ros::Publisher _pub);
	void makePriority(ros::Publisher _pub);
	void makeHealth(ros::Publisher _pub);

public:
	void run(ros::NodeHandle n);
};
