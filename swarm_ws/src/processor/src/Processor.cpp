#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.h"
#include <stdlib.h>
#include <map>
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <tf/LinearMath/Matrix3x3.h>
#include <math.h>
#include <swarm_server/robot_id.h>

wvu_swarm_std_msgs::obs_point_mail Processor::getSeparation(Bot _bot, std::pair<float, float> _obs, float _tolerance) //helper function for finding obstacle points.
{ // takes current bot and looks at distance to each obs point. If it "sees" it, converts that obs to polar and pushes to its vector stored in polar_obs.
	float loc_r; //|distance| b/w bot and current obstacle point.
	float theta; //in radians
	float dx; //x separation.
	float dy; //y separation.
	wvu_swarm_std_msgs::obs_point_mail polar_point;

	dx = _obs.first - _bot.x;
	dy = _obs.second - _bot.y;
	loc_r = sqrt(pow(dx, 2) + pow(dy, 2)); //magnitude of separation
	//std::cout<<"test_tolerance: loc_r = "<<loc_r<<"\n";
	if (loc_r <= _tolerance)
	{
		//std::cout<<"returned polar_point"<<"\n";
		if (dy > 0)
		{
			theta = fmod(atan(dy / dx) - M_PI_2 - _bot.heading, 2 * M_PI);
		} else
		{
			theta = fmod(atan(dy / dx) + M_PI_2 - _bot.heading, 2 * M_PI);
		}
		if (theta < 0)
		{
			theta = 2 * M_PI + theta;
		}

		polar_point.radius = loc_r;
		polar_point.theta = theta;
                return polar_point;
	} else
	{
		polar_point.radius = -1;
		polar_point.theta = 0;
		return polar_point;
	}
}

Processor::Processor(int a) //Default constructor, dummy parameter is there for compile reasons?
{
	{
		for (int i = 0; i < BOT_COUNT; i++)
		{
			bots[i] = Bot();
			activeBots[i]=false;
			for (int j = 0; j < NEIGHBOR_COUNT; j++)
			{
				botMail[i][j] = Bot();
			}
		}
	}

}

Processor::Processor(Bot _bots[], std::pair<float, float> _obs[]) //Constructor given a predetermined set of bots
{
	for (int i = 0; i < BOT_COUNT; i++)
	{
		bots[i] = _bots[i];
	}
	for (int h = 0; h < OBS_POINT_COUNT; h++)
	{
		obs.push_back(_obs[h]);
	}
}
void Processor::init()
{
}

void Processor::processPoints(wvu_swarm_std_msgs::vicon_points data) //Fills in targets
{

	for (size_t i = 0; i < data.point.size(); i++)
	{
		std::pair<float,float> temp(data.point[i].x,data.point[i].y);
		//ROS_INFO("%f, %f", temp.first, temp.second);
                target.push_back(temp);
	}
}

void Processor::processVicon(wvu_swarm_std_msgs::vicon_bot_array data) //Fills in bots[]
{

	for (size_t i = 0; i < data.poseVect.size(); i++)
	{
		char bid[3] = { '\0' };
		bid[0] = data.poseVect[i].botId[0];
		bid[1] = data.poseVect[i].botId[1];
		std::string tempID(bid);

		size_t numID = rid_map[tempID];
		activeBots[numID]=true;
		// the incoming geometry_msgs::Quaternion is transformed to a tf::Quaterion
		tf::Quaternion quat;
		tf::quaternionMsgToTF(data.poseVect[i].botPose.transform.rotation, quat);

		// the tf::Quaternion has a method to access roll pitch and yaw (yaw is all we need in a 2D plane)
		double roll, pitch, yaw;
		tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
		bots[numID] = Bot(tempID, data.poseVect[i].botPose.transform.translation.x,
				data.poseVect[i].botPose.transform.translation.y, yaw, 10000);
	}
}

void Processor::printBots() //Prints the
{
	using namespace std;
	for (int i = 0; i < BOT_COUNT; i++)
	{
		cout << "[" << bots[i].id[0] << bots[i].id[1] << " " << bots[i].x << " " << bots[i].y << " " << bots[i].heading
				<< " ";
		cout << bots[i].distance << "]\n";
	}
}

void Processor::printBotMail() //Prints the id's in botMail[] to the console
{
	using namespace std;
	for (int i = 0; i < BOT_COUNT; i++)
	{
		std::cout << "=========== bots " << i << " Neighbors ========" << "\n";
		std::cout << "[";
		for (int j = 0; j < NEIGHBOR_COUNT; j++)
		{
			cout << " ";
			cout << botMail[i][j].id[0] << botMail[i][j].id[1];
			cout << " ";
		}
		cout << "]\n";
		std::cout << "++++++++++bots " << i << " obstacles++++++" << "\n";

	}
	std::cout << "------------------------" << "\n";
}

void Processor::printAliceMail(wvu_swarm_std_msgs::alice_mail_array _msg)
{
	using namespace std;

	std::cout << "[";
	for (int j = 0; j < NEIGHBOR_COUNT; j++)
	{
		cout << " ";
		cout << _msg.neighborMail[j].theta;
		cout << " ";
	}
	cout << "]\n";

	std::cout << "[";
	for (int j = 0; j < _msg.obsPointMail.size(); j++)
	{
		cout << "|";
		cout << _msg.obsPointMail[j].radius << "," << _msg.obsPointMail[j].theta;
		cout << "|";
	}
	cout << "]\n";
}

/*
 Exactly what it sounds like
 This function finds the nearest few neighbors
 The number of neighbors can be set by NEIGHBOR_COUNT
 */
void Processor::findNeighbors()
{
	int botIndex = 0;
	for (auto &bot : bots)
	{
		int curIndex = 0; // Because we're looping over the array, we have to track the index ourselves
		for (auto &cur : bots)
		{
			if ((cur.id[0] == bot.id[0] && cur.id[1] == bot.id[1])||(activeBots[botIndex]==false && activeBots[curIndex]==false)) // Check for duplicates and nonactive bots
			{
				continue;
			}
			Bot temp(cur);
			temp.distance = sqrt(pow((cur.x - bot.x), 2) + pow((cur.y - bot.y), 2));
			bool smallest = true;

			for (int i = 0; i < NEIGHBOR_COUNT; i++)
			{
				if (temp.distance > botMail[botIndex][i].distance)
				{
					if (i == 0) // If cur is further than the first in the array, it's further
					{
						smallest = false;
						break;
					} else
					{ // This means cur is neither the furthest nor nearest

						botMail[botIndex][0] = temp;
						smallest = false;
						std::sort(botMail[botIndex], botMail[botIndex] + NEIGHBOR_COUNT, compareTwoBots); // Sorts greatest first
						break;
					}
				}
			}
			if (smallest)
			{ // If cur is closer than everything, it's the closest
				botMail[botIndex][0] = temp;
				std::sort(botMail[botIndex], botMail[botIndex] + NEIGHBOR_COUNT, compareTwoBots); // Sorts greatest first
			}
			curIndex++;
		}
		botIndex++;
	}
}

wvu_swarm_std_msgs::neighbor_mail Processor::createNeighborMail(int i, int j)
{

	wvu_swarm_std_msgs::neighbor_mail _neighborMail;
	_neighborMail.id = i;
	if (botMail[i][j].y - bots[i].y > 0)
	{
		_neighborMail.theta = fmod(
				atan((botMail[i][j].y - bots[i].y) / (botMail[i][j].x - bots[i].x)) - M_PI_2 - bots[i].heading, 2 * M_PI);
	} else
	{
		_neighborMail.theta = fmod(
				atan((botMail[i][j].y - bots[i].y) / (botMail[i][j].x - bots[i].x)) + M_PI_2 - bots[i].heading, 2 * M_PI);
	}
	_neighborMail.distance = botMail[i][j].distance;
	_neighborMail.heading = fmod(botMail[i][j].heading - bots[i].heading, 2 * M_PI);


	return _neighborMail;
}

wvu_swarm_std_msgs::obs_point_mail Processor::createTargetMail(int i)
{
	wvu_swarm_std_msgs::obs_point_mail _targetMail;
	int num_pts = target.size();
        for (std::pair<float, float> thisPair : target){
            _targetMail = getSeparation(bots[i], thisPair, 10000);}
	return _targetMail;
}

wvu_swarm_std_msgs::alice_mail_array Processor::createAliceMsg(int i) //Turns information to be sent to Alice into a msg
{

	wvu_swarm_std_msgs::alice_mail_array _aliceMailArray;

	int num_pts = obs.size(); //number of obs pts
	for (int j = 0; j < num_pts; j++)
	{
		wvu_swarm_std_msgs::obs_point_mail _obsPointMail = getSeparation(bots[i], obs.at(j), TOLERANCE);
		if (_obsPointMail.radius > -1)
		{
			_aliceMailArray.obsPointMail.push_back(_obsPointMail);
		}
	}

	for (int j = 0; j < NEIGHBOR_COUNT; j++) //Transfers fields of the struct to fields of the msg
	{
		_aliceMailArray.neighborMail[j] = createNeighborMail(i, j);
	}
	_aliceMailArray.targetMail=createTargetMail(i);
	return _aliceMailArray;

}

void Processor::clearProcessor(){
    target.clear();

}

bool Processor::isActive(int i) //Checks if bot is active
{
	return activeBots[i];
}
