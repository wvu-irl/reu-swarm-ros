#ifndef HUB_H
#define HUB_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/flows.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <wvu_swarm_std_msgs/chargers.h>
#include "alice_swarm/aliceStructs.h"
#include <contour_node/level_description.h>

typedef struct Bot //The Bot struct holds the pose of a robot, along with its distance from another.
{
	Bot() //Default Constructor
	{
		id = -1;
		x = 0;
		y = 0;
		heading = 0;
		distance = 10000;
		swarm_id = -1;
	}

	Bot(int _id, float _x, float _y, float _heading, float _distance,int _sid, ros::Time _time) //Alternate Constructor
	{
		id = _id;

		x = _x;
		y = _y;
		heading = _heading;
		distance = _distance;
		swarm_id=_sid;
		time=_time;
	}

	int id; //the id's are the 50 states, from 0 to 49
	int swarm_id;//which swarm...
	float x; //position
	float y; //position
	float heading; //in radians
	float distance;
	ros::Time time;
} Bot;

bool compareTwoBots(Bot &a, Bot &b) // Reverses the > operator to sort smallest first instead
{
	return (float) a.distance > (float) b.distance;
}

//static const int BOT_COUNT = 50; // Number of bots in the system
static const int NEIGHBOR_COUNT = 4; // Number of neighbors desired
static const int VISION = 40;
//Acts as a parser for the data coming out of the VICON or the simulation, turning the bundle of data into 50 packets
// of individualized information for each swarm bot, allowing them to enact their agent level rules.
class Hub
{

private:
	wvu_swarm_std_msgs::vicon_bot_array viconBotArray;
	wvu_swarm_std_msgs::vicon_points targets;
	wvu_swarm_std_msgs::map_levels map;
	wvu_swarm_std_msgs::flows flows;
	wvu_swarm_std_msgs::chargers chargers;

	std::vector<Bot> bots; //holds locations of all of the bots
	std::vector<std::vector<Bot>> neighbors; //holds locations of all of the closests bots relative to each other

	//Finds the distance between a bot and some object
	std::pair<float,float> getSeparation(Bot _bot, std::pair<float, float> _obs);
	void processVicon(); //Fills in bots[], converst vicon to just the pose data needed
public:

	std::vector<int> ridOrder; //holds the order of the id's of the bots

	Hub(int a); //Default constructor, dummy parameter is there for compile reasons?

	//Adds the msgs gather from various topics to the private fields of Hub
	void update(wvu_swarm_std_msgs::vicon_bot_array &_b, wvu_swarm_std_msgs::vicon_points &_t,
			wvu_swarm_std_msgs::map_levels &_o, wvu_swarm_std_msgs::flows &_f, wvu_swarm_std_msgs::chargers &_c);


	void findNeighbors(); // Finds each robot's nearest neighbors, and thus fills out botMail[]

//	void printAliceMail(wvu_swarm_std_msgs::alice_mail_array _mail); //Prints mail for debug purposes

	void addFlowMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the flows within a robot's VISION range

	void addObsMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the obstacles within a robot's VISION range

	void addTargetMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the targets within a robot's VISION range

	void addNeighborMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Adds the neighbors determined by its closest x

	void addContMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Gives each robot it's value on the contour map

	void addChargerMail(int i, wvu_swarm_std_msgs::alice_mail &_mail); //Gives each robot charger station info

	wvu_swarm_std_msgs::alice_mail_array getAliceMail(); //Gathers all the relative information for a robot into one msg

	void clearHub(); //Clears bot information
};

//#include "Hub.cpp"
#endif
