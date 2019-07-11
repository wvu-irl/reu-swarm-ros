#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <unistd.h>
#include <math.h>

#include <wvu_swarm_std_msgs/chargers.h>


class Hawk_Sim
{
private:
	bool first = true;
	//palce holders for subscription data ---------------
	wvu_swarm_std_msgs::chargers temp_chargers;
	//---------------------------------------------------

	//callback functions
	void chargersCallback(const wvu_swarm_std_msgs::chargers &msg);

	//initializer functions
	void makeChargers(ros::Publisher _pub);

public:
	void run(ros::NodeHandle n);
};
