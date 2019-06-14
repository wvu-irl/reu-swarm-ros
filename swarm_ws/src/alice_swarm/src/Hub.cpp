#include "ros/ros.h"
#include "std_msgs/String.h"
#include "alice_swarm/Hub.h"
#include <stdlib.h>
#include <map>
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <tf/LinearMath/Matrix3x3.h>
#include <math.h>
#include <swarm_server/robot_id.h>


AliceStructs::obj Hub::getSeparation(Bot _bot, std::pair<float, float> _obs, float _tolerance) //helper function for finding obstacle points.
{ // takes current bot and looks at distance to each obs point. If it "sees"[[ it, converts that obs to polar and pushes to its vector stored in polar_obs.
	float loc_r; //|distance| b/w bot and current obstacle point.
	float theta; //in radians
	float dx; //x separation.
	float dy; //y separation.
	AliceStructs::obj polar_point;

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

void Hub::update(wvu_swarm_std_msgs::vicon_bot_array &_b,wvu_swarm_std_msgs::vicon_points &_t,wvu_swarm_std_msgs::vicon_points &_o)
{
	clearHub();
	viconBotArray =_b;
	targets= _t;
	obstacles = _o;
	processVicon(); //needed cause this data needs to be converted first
	findNeighbors();
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
		double roll, pitch, yaw;

		tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
		bots.push_back(Bot(numID, viconBotArray.poseVect[i].botPose.transform.translation.x,
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
	std::cout << "we made it here " << viconBotArray.poseVect.size() << std::endl;
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
				if (temp.distance < it->distance) {
					neighbors.at(botIndex).insert(it,temp);
					done=true;
					std::cout << "we made it here" << std::endl;
				}

				continue;
			}
			if (neighbors.at(botIndex).size() > NEIGHBOR_COUNT) neighbors.at(botIndex).pop_back();
			else if (!done && neighbors.at(botIndex).size()< NEIGHBOR_COUNT) neighbors.at(botIndex).push_back(temp);
		}
	}
}

void Hub::addNeighborMail(int i, AliceStructs::mail &_mail)
{
	std::vector<AliceStructs::neighbor> n;
	for (std::vector<Bot>::iterator it=neighbors.at(i).begin(); it!=neighbors.at(i).end(); it++)
	{

		AliceStructs::neighbor temp;
		temp.name = it->id;
    temp.dir = fmod(atan2(it->y - bots[i].y, it->x - bots[i].x)-bots[i].heading+4*M_PI,2*M_PI);
		temp.dis = it->distance;
		temp.ang = fmod(it->heading - bots[i].heading+2*M_PI, 2 * M_PI);
		n.push_back(temp);
	}
	_mail.neighbors=n;
}

void Hub::addTargetMail(int i, AliceStructs::mail &_mail)
{
	std::vector<AliceStructs::obj> t;
	int num_pts = targets.point.size();
	std::cout << num_pts << std::endl;
	for (int j = 0; j < num_pts; j++)
	{
		std::pair<float, float> temp = {targets.point.at(j).x,targets.point.at(j).y};
		AliceStructs::obj temp2 = getSeparation(bots[i], temp, 10000);
		if (temp2.dis > -1)
		{
			t.push_back(temp2);
		}
	}
	_mail.targets=t;
}

void Hub::addObsPointMail(int i, AliceStructs::mail &_mail)
{
	std::vector<AliceStructs::obj> o;
		int num_pts = obstacles.point.size();
		for (int j = 0; j < num_pts; j++)
		{
			std::pair<float, float> temp = {obstacles.point.at(j).x,obstacles.point.at(j).y};
			AliceStructs::obj temp2 = getSeparation(bots[i], temp, TOLERANCE);
			if (temp2.dis > -1)
			{
				o.push_back(temp2);
			}
		}
		_mail.obstacles=o;
}

AliceStructs::mail Hub::getAliceMail(int i) //Turns information to be sent to Alice into a msg
{
	AliceStructs::mail temp;
	addObsPointMail(i, temp);
	addNeighborMail(i, temp);
	addTargetMail(i, temp);
	temp.name = i;
	return temp;

}

void Hub::clearHub()
{
	bots.clear();
	neighbors.clear();
}
