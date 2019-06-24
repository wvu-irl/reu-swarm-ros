#ifndef HUB_H
#define HUB_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/flows.h>
#include "alice_swarm/aliceStructs.h"

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

	Bot(int _id, float _x, float _y, float _heading, float _distance,int _sid) //Alternate Constructor
	{
		id = _id;

		x = _x;
		y = _y;
		heading = _heading;
		distance = _distance;
		swarm_id=_sid;
	}

	int id; //the id's are the 50 states, from 0 to 49
	int swarm_id;
	float x; //position
	float y; //position
	float heading; //in radians
	float distance;
} Bot;

bool compareTwoBots(Bot &a, Bot &b) // Reverses the > operator to sort smallest first instead
{
	return (float) a.distance > (float) b.distance;
}

//static const int BOT_COUNT = 50; // Number of bots in the system
static const int NEIGHBOR_COUNT = 9; // Number of neighbors desired
static const int VISION = 70;
//Acts as a parser for the data coming out of the VICON or the simulation, turning the bundle of data into 50 packets
// of individualized information for each swarm bot, allowing them to enact their agent level rules.
class Hub
{

private:
	wvu_swarm_std_msgs::vicon_bot_array viconBotArray;
	wvu_swarm_std_msgs::vicon_points targets;
	wvu_swarm_std_msgs::vicon_points obstacles;
	wvu_swarm_std_msgs::flows flows;

	std::vector<Bot> bots; //holds locations of all of the bots
	std::vector<std::vector<Bot>> neighbors; //holds locations of all of the closests bots relative to each other
	std::vector<int> ridOrder; //holds the order of the id's of the bots

	//Finds the distance between a bot and some object
	AliceStructs::obj getSeparation(Bot _bot, std::pair<float, float> _obs, float _tolerance);
	void processVicon(); //Fills in bots[], converst vicon to just the pose data needed
public:
	Hub(int a); //Default constructor, dummy parameter is there for compile reasons?

	//Adds the msgs gathere from various topics to the private fields of Hub
	void update(wvu_swarm_std_msgs::vicon_bot_array &_b, wvu_swarm_std_msgs::vicon_points &_t,
			wvu_swarm_std_msgs::vicon_points &_o, wvu_swarm_std_msgs::flows &_f);


	void findNeighbors(); // Finds each robot's nearest neighbors, and thus fills out botMail[]

	void printAliceMail(AliceStructs::mail _mail); //Prints mail for debug purposes

	void addFlowMail(int i, AliceStructs::mail &_mail); //Adds the flows within a robot's VISION range

	void addObsPointMail(int i, AliceStructs::mail &_mail); //Adds the obstacles within a robot's VISION range

	void addTargetMail(int i, AliceStructs::mail &_mail); //Adds the targets within a robot's VISION range

	void addNeighborMail(int i, AliceStructs::mail &_mail); //Adds the neighbors determined by its closest x

	AliceStructs::mail getAliceMail(int i); //Gathers all the relative information for a robot into one struct

	void clearHub(); //Clears bot information
};

//#include "Hub.cpp"
#endif
