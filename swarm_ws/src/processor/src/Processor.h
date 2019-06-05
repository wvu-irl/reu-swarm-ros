#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/aliceMail.h>
#include <wvu_swarm_std_msgs/aliceMailArray.h>
#include <wvu_swarm_std_msgs/viconBotArray.h>

typedef struct Bot //The Bot struct holds the pose of a robot, along with its distance from another.
{
	Bot() //Default Constructor
	{
		id[0] = '\0';
		id[1] = '\0';
		x = 0;
		y = 0;
		heading = 0;
		distance = 10000;
	}

	Bot(float _x, float _y, std::string name) //Alternate Constructor
	{
		x = _x;
		y = _y;
		strcpy(id, name.c_str());
		heading = 0;
		distance = 10000;
	}

	Bot(char _id[2], float _x, float _y, float _heading, float _distance) //Alternate Constructor
	{
		id[0] = _id[0];
		id[1] = _id[1];
		x = _x;
		y = _y;
		heading = _heading;
		distance = _distance;
	}

	char id[2]; //the id's are the 50 states, in char[2] form.
	float x; //position
	float y; //position
	float heading; //in radians
	float distance; //is squared
} Bot;

typedef struct Obstacle //holds two vectors, which concurrently house a point cloud.
{
	Obstacle() //Default Constructor
	{
	}

	Obstacle(std::vector<float> _x, std::vector<float> _y) //Alternate Constructor
	{
		std::vector<float> x = _x;
		std::vector<float> y = _y;
	}

	std::vector<float> x;
	std::vector<float> y;
} Obstacle;

bool compareTwoBots(Bot &a, Bot &b) // Reverses the > operator to sort smallest first instead
{
	return (float) a.distance > (float) b.distance;
}

std::map<char *, int> inst() // creates a map for all the string to integer values
{
	char *LETTERARRAY[] =
	{ "DE", "PA", "NJ", "GA", "CT", "MA", "MD", "SC", "NH", "VA", "NY", "NC",
			"RI", "VT", "KY", "TN", "OH", "LA", "IN", "MS", "IL", "AL", "ME", "MO",
			"AR", "MI", "FL", "TX", "IA", "WI", "CA", "MN", "OR", "KA", "WV", "NV",
			"NE", "CO", "ND", "SD", "MT", "WA", "ID", "WY", "UT", "OK", "NM", "AZ",
			"AK", "HI" };
	std::map<char *, int> map;
	for (size_t i = 0; i < sizeof(LETTERARRAY) / sizeof(LETTERARRAY[0]); i++)
	{
		map.insert(std::pair<char *, size_t>(LETTERARRAY[i], i));
	}
	return map;
}

std::map<char *, int> rid_map = inst(); //a map of the states to integers

static const int BOT_COUNT = 10; // Number of bots in the system
static const int NEIGHBOR_COUNT = 4; // Number of neighbors desired
static const int OBS_POINT_COUNT = 4; //random number of obstacles

//Acts as a parser for the data coming out of the VICON or the simulation, turning the bundle of data into 50 packets
// of individualized information for each swarm bot, allowing them to enact their agent level rules.
class Processor
{

private:
	Bot bots[BOT_COUNT]; //Stores the information from the VICON
	Bot botMail[BOT_COUNT][NEIGHBOR_COUNT]; //Stores the information to be sent to Alice

	std::vector<std::pair<float, float>> obs; //vector of all obstacle points
	std::vector<std::pair<float, float>> polar_obs[BOT_COUNT]; //Array of vectors of pairs.
	//each bot has a vector of obs pairs it can "see". Pairs are form (r,theta).
	ros::Timer timer;
	ros::Publisher pub;
	ros::Subscriber sub;

	std::pair<float, float> getSeparation(Bot _bot, std::pair<float, float> _obs,
			float _tolerance);

public:
	Processor(int a); //Default constructor, dummy parameter is there for compile reasons?

	Processor(Bot _bots[], std::pair<float, float> _obs[]); //Constructor given a predetermined set of bots

	void init(); //Does nothing for now

	void processVicon(wvu_swarm_std_msgs::viconBotArray data); //Fills in bots[]

	void printBotMail(); //Prints botMail[] to the console

	void printAliceMail(wvu_swarm_std_msgs::aliceMailArray msg);

	void findNeighbors(); // Finds each robot's nearest neighbors, and thus fills out botMail[]

	wvu_swarm_std_msgs::aliceMailArray createAliceMsg(int i); //Turns information to be sent to Alice into a msg
};

//#include "Processor.cpp"
#endif

