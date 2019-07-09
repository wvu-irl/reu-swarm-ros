#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <unistd.h>
#include <math.h>

#include <wvu_swarm_std_msgs/chargers.h>


class Hawk_Sim_Setup
{
private:
	//callback functions
	void chargersCallback(wvu_swarm_std_msgs::chargers &msg)

	//initializer functions
	void makeChargers(ros::Publisher _pub);

public:
	void Hawk_Sim_Setup::run();
};
