#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_points.h>

typedef struct Bot //The Bot struct holds the pose of a robot, along with its distance from another.
{
	Bot() //Default Constructor
	{
		id = -1;
		x = 0;
		y = 0;
		heading = 0;
		distance = 10000;
	}

	Bot(float _x, float _y, int _id) //Alternate Constructor
	{
		x = _x;
		y = _y;
		id = _id;
		heading = 0;
		distance = 10000;

	}

	Bot(int _id, float _x, float _y, float _heading, float _distance) //Alternate Constructor
	{
		id=_id;

		x = _x;
		y = _y;
		heading = _heading;
		distance = _distance;
	}

	int id; //the id's are the 50 states, from 0 to 49
	float x; //position
	float y; //position
	float heading; //in radians
	float distance;
} Bot;

bool compareTwoBots(Bot &a, Bot &b) // Reverses the > operator to sort smallest first instead
{
	return (float) a.distance > (float) b.distance;
}

static const int BOT_COUNT = 50; // Number of bots in the system
static const int NEIGHBOR_COUNT = 4; // Number of neighbors desired
static const int OBS_POINT_COUNT = 4; //random number of obstacles
static const int TOLERANCE = 12;
//Acts as a parser for the data coming out of the VICON or the simulation, turning the bundle of data into 50 packets
// of individualized information for each swarm bot, allowing them to enact their agent level rules.
class Processor
{

private:
	Bot bots[BOT_COUNT]; //Stores the information from the VICON
	Bot botMail[BOT_COUNT][NEIGHBOR_COUNT]; //Stores the information to be sent to Alice
	bool activeBots[BOT_COUNT];
	std::vector<std::pair<float, float>> obs; //vector of all obstacle points
	std::vector<std::pair<float, float>> target;
	//each bot has a vector of obs pairs it can "see". Pairs are form (r,theta).
	ros::Timer timer;
	ros::Publisher pub;
	ros::Subscriber sub;

	wvu_swarm_std_msgs::obs_point_mail getSeparation(Bot _bot, std::pair<float, float> _obs, float _tolerance);

public:
	Processor(int a); //Default constructor, dummy parameter is there for compile reasons?

	Processor(Bot _bots[], std::pair<float, float> _obs[]); //Constructor given a predetermined set of bots

	void init(); //Does nothing for now

	void processVicon(wvu_swarm_std_msgs::vicon_bot_array data); //Fills in bots[]

	void processPoints(wvu_swarm_std_msgs::vicon_points data);

	void printBots();

	void printBotMail(); //Prints botMail[] to the console

	void printAliceMail(wvu_swarm_std_msgs::alice_mail_array msg); //prints alice_mail_array

	void findNeighbors(); // Finds each robot's nearest neighbors, and thus fills out botMail[]

	void addObsPointMail(int i,wvu_swarm_std_msgs::alice_mail_array &_aliceMailArray);

	void addTargetMail(int i,wvu_swarm_std_msgs::alice_mail_array &_aliceMailArray);

	void addNeighborMail(int i, wvu_swarm_std_msgs::alice_mail_array &_aliceMailArray);//Creates a neighbor_mail msg

	wvu_swarm_std_msgs::alice_mail_array createAliceMsg(int i); //Compiles all info into a single msg

	bool isActive(int i);

	void clearProcessor();
};

//#include "Processor.cpp"
#endif
