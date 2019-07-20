#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <swarm_server/battery_states.h>
#include <wvu_swarm_std_msgs/chargers.h>
#include <wvu_swarm_std_msgs/priorities.h>
#include <wvu_swarm_std_msgs/energy.h>
#include <wvu_swarm_std_msgs/sensor_data.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>


class Hawk_Sim
{
private:
	float counter = 0;
	float prev_counter = 1;

	int NUMBOTS;
	bool first = true;
	bool energy_first = true;
	bool new_chargers;
	//palce holders for subscription data ---------------
	wvu_swarm_std_msgs::chargers temp_chargers;
	wvu_swarm_std_msgs::chargers prev_temp_chargers;
	float prev_rid = -1;
	wvu_swarm_std_msgs::priorities temp_priorities;
	wvu_swarm_std_msgs::energy temp_energy;
	wvu_swarm_std_msgs::sensor_data temp_sd;
	wvu_swarm_std_msgs::vicon_bot_array temp_bots;
	//---------------------------------------------------

	//callback functions
	void chargersCallback(const wvu_swarm_std_msgs::chargers &msg);
	void priorityCallback(const wvu_swarm_std_msgs::priorities &msg);
	void energyCallback(const wvu_swarm_std_msgs::energy &msg);
	void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg);

	//initializer functions
	void makeChargers(ros::Publisher _pub);
	void makePriority(ros::Publisher _pub);
	void makeEnergy(ros::Publisher _pub);
	void makeSensorData(ros::Publisher _pub);

public:
	void run(ros::NodeHandle n);
};
