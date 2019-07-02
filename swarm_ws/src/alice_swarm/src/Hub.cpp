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
#define DEBUG_HUB 0

std::pair<float, float> Hub::getSeparation(Bot _bot, std::pair<float, float> _obs) //helper function for finding obstacle points.
{ // takes current bot and looks at distance to each obs point. If it "sees"[[ it, converts that obs to polar and pushes to its vector stored in polar_obs.
	float loc_r; //|distance| b/w bot and current obstacle point.
	float theta; //in radians
	float dx; //x separation.
	float dy; //y separation.
	std::pair<float, float> to_return;

	dx = _obs.first - _bot.x;
	dy = _obs.second - _bot.y;
	loc_r = sqrt(pow(dx, 2) + pow(dy, 2)); //magnitude of separation
	theta = fmod(atan2(dy, dx) - _bot.heading, 2 * M_PI);
	to_return.first = loc_r * cos(theta);
	to_return.second = loc_r * sin(theta);
	return to_return;
}

Hub::Hub(int a) //Default constructor, dummy parameter is there for compile reasons?
{
}

void Hub::update(wvu_swarm_std_msgs::vicon_bot_array &_b, wvu_swarm_std_msgs::vicon_points &_t,
		wvu_swarm_std_msgs::map_levels &_o, wvu_swarm_std_msgs::flows &_f)
{
	clearHub();
	viconBotArray = _b;
	targets = _t;
	map = _o;
	flows = _f;
	processVicon(); //needed cause this data needs to be converted first
	findNeighbors();
}

void Hub::processVicon() //Fills in bots[]
{

	for (size_t i = 0; i < viconBotArray.poseVect.size(); i++)
	{
		//This char bs-ery is for converting the state initials to numbers in our map
		char bid[3] =
		{ '\0' };
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
		bots.push_back(
				Bot(numID, viconBotArray.poseVect[i].botPose.transform.translation.x,
						viconBotArray.poseVect[i].botPose.transform.translation.y, yaw, 10000, i % 2 + 1,
						viconBotArray.poseVect[i].botPose.header.stamp));
		std::vector<Bot> temp;
		ridOrder.push_back(numID); //storing the order of insertion
		neighbors.push_back(temp); //adds an empty vector to neighbors for future use
	}
}

/*
 Exactly what it sounds like
 This function finds the nearest few neighbors
 The number of neighbors can be set by NEIGHBOR_COUNT
 */
void Hub::findNeighbors()
{
	for (int botIndex = 0; botIndex < viconBotArray.poseVect.size(); botIndex++)
	{
		for (int curIndex = 0; curIndex < viconBotArray.poseVect.size(); curIndex++)
		{
			if (botIndex == curIndex) // Check for duplicates
			{
				continue;
			}
			Bot temp(bots.at(curIndex));

			//Finds the distance between two bots
			temp.distance = sqrt(
					pow((bots.at(curIndex).x - bots.at(botIndex).x), 2) + pow((bots.at(curIndex).y - bots.at(botIndex).y), 2));

			bool done = false; //keeps track of whether or not the bot has been inserted as a neighbor
			for (std::vector<Bot>::iterator it = neighbors.at(botIndex).begin(); it != neighbors.at(botIndex).end(); it++)
			{
				if (temp.distance < it->distance) //Checks if the current bot is closer than the bot stored as a neighbor
				{
					neighbors.at(botIndex).insert(it, temp);
					done = true;
					break;
				}
			}
			if (neighbors.at(botIndex).size() > NEIGHBOR_COUNT) //If there are too many neighbors, the furthest one is discarded
			{
				neighbors.at(botIndex).pop_back();

			} //If there are too few neighbors and the current on hasn't been inserted, it is made a neighbor
			else if (!done && neighbors.at(botIndex).size() < NEIGHBOR_COUNT)
				neighbors.at(botIndex).push_back(temp);
		}
	}
}

void Hub::addNeighborMail(int i, wvu_swarm_std_msgs::alice_mail &_mail)
{
	for (std::vector<Bot>::iterator it = neighbors.at(i).begin(); it != neighbors.at(i).end(); it++)
	{
		wvu_swarm_std_msgs::neighbor_mail temp;
		temp.name = it->id;
		//Makes the direction of the neighbor relative to the robot's heading
		float loc_r = it->distance; //magnitude of separation
		float theta = fmod(atan2(it->y - bots[i].y, it->x - bots[i].x) - bots[i].heading + 4 * M_PI, 2 * M_PI);
		temp.x = loc_r * cos(theta);
		temp.y = loc_r * sin(theta);
		//Makes the heading of the neighbor relative to the robot's heading
		temp.ang = fmod(it->heading - bots[i].heading + 2 * M_PI, 2 * M_PI);
		temp.sid = it->swarm_id;
		_mail.neighborMail.push_back(temp);
	}
}

void Hub::addFlowMail(int i, wvu_swarm_std_msgs::alice_mail &_mail)
{
	int num_pts = flows.flow.size();
	for (int j = 0; j < num_pts; j++)
	{
		if (flows.flow.at(j).sid == bots[i].swarm_id || flows.flow.at(j).sid == 0)
		{
			std::pair<float, float> temp =
			{ flows.flow.at(j).x, flows.flow.at(j).y };
			std::pair<float, float> temp2 = getSeparation(bots[i], temp);
			if (pow(pow(temp2.first, 2) + pow(temp2.second, 2), 0.5) < VISION) //If the flow was in VISION range
			{
				wvu_swarm_std_msgs::flow_mail temp3;
				temp3.x = temp2.first;
				temp3.y = temp2.second;
				//Makes the direction of the flow relative to the robot's heading
				temp3.dir = fmod(flows.flow.at(j).theta - bots[i].heading + 2 * M_PI, 2 * M_PI);
				temp3.spd = flows.flow.at(j).r;
				temp3.pri = flows.flow.at(j).pri;
				_mail.flowMail.push_back(temp3);
			}
		}
	}
}

void Hub::addTargetMail(int i, wvu_swarm_std_msgs::alice_mail &_mail) //Adds targets within a robots vision range
{
	int num_pts = targets.point.size();
	for (int j = 0; j < num_pts; j++)
	{
		std::pair<float, float> temp =
		{ targets.point.at(j).x, targets.point.at(j).y };
		std::pair<float, float> temp2 = getSeparation(bots[i], temp);
		if (pow(pow(temp2.first, 2) + pow(temp2.second, 2), 0.5) < VISION) //If the target was in VISION range
		{
			wvu_swarm_std_msgs::point_mail temp3;
			temp3.x = temp2.first;
			temp3.y = temp2.second;
			_mail.targetMail.push_back(temp3);
		}
	}
}

void Hub::addObsMail(int i, wvu_swarm_std_msgs::alice_mail &_mail) //Adds obstacles within a robots vision range
{
	if (map.levels.size() > map_ns::OBSTACLE)
	{
		int num_pts = map.levels.at(map_ns::OBSTACLE).functions.size();
		for (int j = 0; j < num_pts; j++)
		{

			wvu_swarm_std_msgs::ellipse temp3 = map.levels.at(map_ns::OBSTACLE).functions.at(j).ellipse;
			std::pair<float, float> temp =
			{ temp3.offset_x, temp3.offset_y };
			std::pair<float, float> temp2 = getSeparation(bots[i], temp);
			if (pow(pow(temp2.first, 2) + pow(temp2.second, 2), 0.5) < VISION)
			{
				temp3.offset_x = temp2.first;
				temp3.offset_y = temp2.second;
				temp3.theta_offset = fmod(temp3.theta_offset - bots[i].heading + 2 * M_PI, 2 * M_PI);
				_mail.obsMail.push_back(temp3);
			}
		}
	}
}

void Hub::addContMail(int i, wvu_swarm_std_msgs::alice_mail &_mail) //Gives each robot it's value on the contour map
{
	wvu_swarm_std_msgs::vicon_point loc;
	loc.x = bots[i].x;
	loc.y = bots[i].y;
	std::cout << loc.x << " "<< loc.y << std::endl;
	if (map.levels.size() > map_ns::TARGET)
	{
		_mail.contourVal = (float) map_ns::calculate(map.levels.at(map_ns::TARGET), loc);
	} else
		_mail.contourVal = 0;
}

//void Hub::printAliceMail(wvu_swarm_std_msgs::alice_mail _mail) //Prints mail for debug purposes
//{
//	std::cout << "--- Mail for Alice " << _mail.name << "," << _mail.sid << " ---" << std::endl;
//	std::cout << "Neighbors - " << _mail.neighbors.size() << std::endl;
//	for (std::vector<AliceStructs::neighbor>::iterator it = _mail.neighbors.begin(); it != _mail.neighbors.end(); ++it)
//	{
//
//		std::cout << it->dir << " " << it->dis << " " << it->ang << " " << it->name << std::endl;
//	}
////	std::cout << "Obstacles - " << _mail.obstacles.size() << std::endl;
////	for (std::vector<AliceStructs::obj>::iterator it = _mail.obstacles.begin(); it != _mail.obstacles.end(); ++it)
////	{
////
////		std::cout << it->dir << " " << it->dis << " " << it->ang << " " << it->name << std::endl;
////	}
//	//will finish if necessary but seems unnecessary...
//}

wvu_swarm_std_msgs::alice_mail_array Hub::getAliceMail() //Gathers all the relative information for a robot into one msg
{
	wvu_swarm_std_msgs::alice_mail_array to_return;
	for (std::vector<int>::iterator it = ridOrder.begin(); it != ridOrder.end(); ++it)
	{
		wvu_swarm_std_msgs::alice_mail temp;
		addObsMail(*it, temp);
		addNeighborMail(*it, temp);
		addTargetMail(*it, temp);
		addFlowMail(*it, temp);
		addContMail(*it, temp);
		temp.name = *it;
		temp.sid = bots[*it].swarm_id;
		temp.time = bots[*it].time;
		temp.x= bots[*it].x;
		temp.y=bots[*it].y;
		temp.heading=bots[*it].heading;
		to_return.mails.push_back(temp);

	}
#if DEBUG_HUB
	//printAliceMail(temp);
#endif
	return to_return;

}

void Hub::clearHub() //Clears information about the robots
{
	ridOrder.clear();
	bots.clear();
	neighbors.clear();
}
