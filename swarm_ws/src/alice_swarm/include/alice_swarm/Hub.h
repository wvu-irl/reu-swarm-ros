#ifndef HUB_H
#define HUB_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/neighbor_mail.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
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

//static const int BOT_COUNT = 50; // Number of bots in the system
static const int NEIGHBOR_COUNT = 4; // Number of neighbors desired
static const int OBS_POINT_COUNT = 4; //random number of obstacles
static const int TOLERANCE = 100;
//Acts as a parser for the data coming out of the VICON or the simulation, turning the bundle of data into 50 packets
// of individualized information for each swarm bot, allowing them to enact their agent level rules.
class Hub
{

private:
	wvu_swarm_std_msgs::vicon_bot_array viconBotArray;
	wvu_swarm_std_msgs::vicon_points targets;
	wvu_swarm_std_msgs::vicon_points obstacles;
	//each bot has a vector of obs pairs it can "see". Pairs are form (r,theta).
//	ros::Timer timer;
//	ros::Publisher pub;
//	ros::Subscriber sub;

	std::vector<Bot> bots;
	std::vector<std::vector <Bot>> neighbors;
	std::vector<int> ridOrder;
	AliceStructs::obj getSeparation(Bot _bot, std::pair<float, float> _obs, float _tolerance);
	void processVicon(); //Fills in bots[]
public:
	Hub(int a); //Default constructor, dummy parameter is there for compile reasons?

	void update(wvu_swarm_std_msgs::vicon_bot_array &_b,wvu_swarm_std_msgs::vicon_points &_t,wvu_swarm_std_msgs::vicon_points &_o);

	void findNeighbors(); // Finds each robot's nearest neighbors, and thus fills out botMail[]

	void printAliceMail(AliceStructs::mail _mail);


	void addObsPointMail(int i, AliceStructs::mail &_mail);

	void addTargetMail(int i, AliceStructs::mail &_mail);

	void addNeighborMail(int i, AliceStructs::mail &_mail);//Creates a neighbor_mail msg

	AliceStructs::mail getAliceMail(int i); //Compiles all info into a single msg

	void clearHub();
};

//#include "Hub.cpp"
#endif
