#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Hub.h"
#include <stdlib.h>
#include <map>
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <tf/LinearMath/Matrix3x3.h>
#include <math.h>
#include <swarm_server/robot_id.h>

aliceStructs::tar Hub::getSeparation(Bot _bot, std::pair<float, float> _obs, float _tolerance) //helper function for finding obstacle points.
{ // takes current bot and looks at distance to each obs point. If it "sees" it, converts that obs to polar and pushes to its vector stored in polar_obs.
	float loc_r; //|distance| b/w bot and current obstacle point.
	float theta; //in radians
	float dx; //x separation.
	float dy; //y separation.
	aliceStructs::tar polar_point;

	dx = _obs.first - _bot.x;
	dy = _obs.second - _bot.y;
	loc_r = sqrt(pow(dx, 2) + pow(dy, 2)); //magnitude of separation
	//std::cout<<"test_tolerance: loc_r = "<<loc_r<<"\n";
	if (loc_r <= _tolerance)
	{
                theta = fmod(atan2(dy, dx)-_bot.heading+4*M_PI,2*M_PI);
		polar_point.dis = loc_r;
		polar_point.dir = theta;
//		polar_point.radius = 2;
//		polar_point.theta = 2;
		return polar_point;
	} else
	{
		polar_point.dis = -1;
		polar_point.dir = 0;
		return polar_point;
	}
}

Hub::Hub(int a) //Default constructor, dummy parameter is there for compile reasons?
{
}

Hub::update(wvu_swarm_std_msgs::vicon_bot_array &_b,wvu_swarm_std_msgs::vicon_points &_t,wvu_swarm_std_msgs::vicon_points &_o)
{
	viconBotArray =_b;
	targets= _t;
	obstacles = _o;
	processVicon(); //needed cause this data needs to be converted first
}


void Hub::processVicon() //Fills in bots[]
{

	for (size_t i = 0; i < viconBotArray.poseVect.size(); i++)
	{
		char bid[3] = { '\0' };
		bid[0] = viconBotArray.poseVect[i].botId[0];
		bid[1] = viconBotArray.poseVect[i].botId[1];
		std::string tempID(bid);
		size_t numID = rid_map[tempID];
		// the incoming geometry_msgs::Quaternion is transformed to a tf::Quaterion
		tf::Quaternion quat;
		tf::quaternionMsgToTF(viconBotArray.poseVect[i].botPose.transform.rotation, quat);

		// the tf::Quaternion has a method to access roll pitch and yaw (yaw is all we need in a 2D plane)
		double roll, pitch, yaw;std::list <neighbor> bots, float tolerance
		tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
		bots.insert(Bot(numID, viconBotArray.poseVect[i].botPose.transform.translation.x,
				viconBotArray.poseVect[i].botPose.transform.translation.y, yaw, 10000));
	}
}

/*
 Exactly what it sounds like
 This function finds the nearest few neighbors
 The number of neighbors can be set by NEIGHBOR_COUNT
 */
void Hub::findNeighbors()
{
	for (int botIndex=0; botIndex<viconBotArray.poseVect.size(); botIndex++)
	{
		for (int curIndex=0; curIndex<viconBotArray.poseVect.size(); curIndex++)
		{
			if (botIndex==curIndex) // Check for duplicates and nonactive bots
			{
				continue;
			}
			Bot temp(bots.at(curIndex));
			temp.distance = sqrt(pow((bots.at(curIndex).x - bots.at(botIndex).x), 2) + pow((bots.at(curIndex).y - bots.at(botIndex).y), 2));
			bool done = false;
			for (std::vector<Bot>::iterator it=neighbors.at(botIndex).begin(); it!=neighbors.at(botIndex).end(); ++it)
			{
				if (temp.distance < it->distance) neightbors.at(botIndex).insert(it,temp);
				done=true;
				continue;
			}
			if (neighbors.at(botIndex).size() > NEIGHBOR_COUNT) neighbors.at(botIndex).pop_back();
			else if (!done && neighbors.at(botIndex).size()< NEIGHBOR_COUNT) neighbors.at(botIndex).push_back(temp);
		}
	}
}

void Hub::addNeighborMail(int i, aliceStructs::mail &_mail)
{
	std::vector<aliceStructs::neighbor> n;
	for (std::vector<Bot>::iterator it=neighbors.at(i).begin(); it!=neighbors.at(i).end(); ++it)
	{
		aliceStructs::neighbor temp;
		temp.name = it->id;
    temp.dir = fmod(atan2(it->y - bots[i].y, it->x - bots[i].x)-bots[i].heading+4*M_PI,2*M_PI);
		temp.dis = it->distance;
		temp.ang = fmod(it->heading - bots[i].heading+2*M_PI, 2 * M_PI);
		n.push_back(temp);
	}
	mail.neighbors=n;
}

void Hub::addTargetMail(int i, AliceStructs::mail &_mail)
{
	std::vector<aliceStructs::tar> t;
//		for (std::vector<Bot>::iterator it=neighbors.at(i).begin(); it!=neighbors.at(i).end(); ++it)
//		{
//			aliceStructs::neighbor temp;
//			temp.name = it->id;
//	    temp.dir = fmod(atan2(it->y - bots[i].y, it->x - bots[i].x)-bots[i].heading+4*M_PI,2*M_PI);
//			temp.dis = it->distance;
//			temp.ang = fmod(it->heading - bots[i].heading+2*M_PI, 2 * M_PI);
//			n.push_back(temp);
//		}
//		mail.neighbors=n;
	int num_pts = targets.size();
	for (int j = 0; j < num_pts; j++)
	{
		aliceStructs::tar temp = getSeparation(bots[i], target.at(j), 10000);
		if (_obsPointMail.radius > -1)
		{
			t.push_back(temp);
		}
	}
}

void Hub::addObsPointMail(int i, AliceStructs::mail &_mail)
{
	int num_pts = obs.size(); //number of obs pts
	for (int j = 0; j < num_pts; j++)
	{
		wvu_swarm_std_msgs::obs_point_mail _obsPointMail = getSeparation(bots[i], obs.at(j), TOLERANCE);
		if (_obsPointMail.radius > -1)
		{
			_aliceMailArray.obsPointMail.push_back(_obsPointMail);
		}
	}
}

AliceStructs::mail Hub::getAliceMail(int i) //Turns information to be sent to Alice into a msg
{
	AliceStructs::mail temp;
	addObsPointMail(i, temp);
	addNeighborMail(i, temp);
	addTargetMail(i, temp);
	name = i;
	return temp;

}

void Hub::clearHub()
{
        for(int i= 0; i<BOT_COUNT; i++)
        {
            for(int j=0; j<NEIGHBOR_COUNT; j++) 
            {
                botMail[i][j].id=-1;
                botMail[i][j].distance=10000;
            }
        }
}

bool Hub::isActive(int i) //Checks if bot is active
{
return activeBots[i];
}
