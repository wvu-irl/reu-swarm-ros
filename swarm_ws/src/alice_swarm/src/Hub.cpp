/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

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

#define DEBUG_HUB 1
#define DEBUG_ChargerMail 0

Hub::Hub(int a) //Default constructor, dummy parameter is there for compile reasons?
{
}

std::pair<float, float> Hub::getSeparation(Bot _bot, std::pair<float, float> _obs)
{
	float loc_r; //|distance| b/w bot and current obstacle point.
	float theta; //in radians
	float dx; //x separation.
	float dy; //y separation.
	std::pair<float, float> to_return;

	dx = _obs.first - _bot.x;
	dy = _obs.second - _bot.y;
	loc_r = sqrt(pow(dx, 2) + pow(dy, 2)); //magnitude of separation
	theta = fmod(atan2(dy, dx) - _bot.heading, 2 * M_PI);
	to_return.first = loc_r * cos(theta); //x in cur frame
	to_return.second = loc_r * sin(theta); //y in cur frame
	return to_return;
}

void Hub::updateSensorDatas()
{
	if (ridOrder.size() != sensor_datas.size()) //if the number of bots and number of received packets don't match up
	{
		bool init = false;
		int i = 0;
		while (i < sensor_datas.size()) //checks if this sensor data is already in the vector.
		{
			if (sensor_data.rid == sensor_datas.at(i).rid)
			{
				sensor_datas.at(i) = sensor_data;
				i = sensor_datas.size() + 1;
				init = true;
			}
			i++;
		}
		if (!init && validRID()) //if the id matches a bot in the sim, and the sensor_data is not in senor_datas
		{
			std::vector<wvu_swarm_std_msgs::sensor_data> temp_sds;
			for (int i = 0; i < sensor_datas.size(); i++) //copies sensor_datas into a temperary structure for sorting.
			{
				temp_sds.push_back(sensor_datas.at(i));
			}
			sensor_datas.push_back(sensor_data); //makes both vectors the right size.
			temp_sds.push_back(sensor_data);

			int temp_index = 0;
			for (int i = 0; i < ridOrder.size(); i++) //sensor_datas in same order as ridOrder. temp_sds is used as a place holder for the old sensor_datas vecoter.
			{
				for (int j = 0; j < temp_sds.size(); j++)
				{
					if (ridOrder.at(i) == temp_sds.at(j).rid)
					{
						sensor_datas.at(temp_index) = temp_sds.at(j);
						temp_index++;
					}
				}
			}
		}
	} else //check more matching id in sensor_datas, replace that data if matched. If no match does nothing.
	{
		for (int i = 0; i < sensor_datas.size(); i++)
		{
			if (sensor_datas.at(i).rid == sensor_data.rid)
			{
				sensor_datas.at(i) = sensor_data;
			}
		}
	}
}

bool Hub::validRID() //returns if the id of the current sensor_data message is in the sensor_datas vector.
{
	bool result = false;
	for (int i = 0; i < ridOrder.size(); i++)
	{
		if (ridOrder.at(i) == sensor_data.rid) //checks if the sensor's id matches a robot in operation.
		{
			result = true;
		}
	}
	return result;
}

void Hub::processVicon()
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

		// a check to find if the robot has been seen before
		bool foundID = false;
		int j;
		for (j = 0; j < ridOrder.size(); j++)
		{
			if (ridOrder.at(j) == numID)
			{
				foundID = true;
				break;
			}
		}

		if (foundID) //if the robot has been seen before...
		{
			//simply replace the old bot struct with the one with new information
			bots.at(j) = Bot(numID, viconBotArray.poseVect[i].botPose.transform.translation.x,
					viconBotArray.poseVect[i].botPose.transform.translation.y, yaw, 10000, numID % 2 + 1,
					viconBotArray.poseVect[i].botPose.header.stamp);

		} else
		{
			//otherwise insert a new bot within the vector
			bots.push_back(
					Bot(numID, viconBotArray.poseVect[i].botPose.transform.translation.x,
							viconBotArray.poseVect[i].botPose.transform.translation.y, yaw, 10000, numID % 2 + 1,
							viconBotArray.poseVect[i].botPose.header.stamp));

			ridOrder.push_back(numID); //record the insertion of this new robot
		}
	}
}

void Hub::findNeighbors()
{
	//neighbors is empty prior to this
	for (int i = 0; i < ridOrder.size(); i++)
	{
		std::vector<Bot> temp;
		neighbors.push_back(temp); //adds an empty vector to neighbors for future use
	}

	for (int botIndex = 0; botIndex < ridOrder.size(); botIndex++) //iterate through list of active robots
	{
		for (int curIndex = 0; curIndex < ridOrder.size(); curIndex++) //iterate through list of active robots
		{
			if (botIndex == curIndex) // Check for duplicates
			{
				continue;
			}

			Bot temp(bots.at(curIndex)); //make a copy of the bot at cur

			//Finds the distance between two bots
			temp.distance = sqrt(
					pow((bots.at(curIndex).x - bots.at(botIndex).x), 2) + pow((bots.at(curIndex).y - bots.at(botIndex).y), 2));

			if (temp.distance > VISION * 2) continue; //if the robot is outside of vision range, exit. Vision range for other bots was enlarged

			bool done = false; //keeps track of whether or not the bot has been inserted as a neighbor

			for (std::vector<Bot>::iterator it = neighbors.at(botIndex).begin(); it != neighbors.at(botIndex).end(); it++)//iterate throught the list of known neighbors
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
	if (map.levels.size() > map_ns::TARGET)
	{
		_mail.contVal = (float) map_ns::calculate(map.levels.at(map_ns::TARGET), loc);
	} else
		_mail.contVal = 0;
}

void Hub::addChargerMail(int i, wvu_swarm_std_msgs::alice_mail &_mail) //converts each charger into relative coordinates of the current bot. Adds it to mail.
{
#if DEBUG_ChargerMail
	std::cout<<"############### FROM THE HUB ################"<<std::endl;
#endif
	for (int j = 0; j < chargers.charger.size(); j++)
	{
		//makes the x coord that of the way point of that charger, for charge() rule. (Needs to be 5 cm in front).
		std::pair<float, float> temp_t =
		{ chargers.charger.at(j).x + 10, chargers.charger.at(j).y };
		std::pair<float, float> temp_a =
		{ chargers.charger.at(j).x + 3, chargers.charger.at(j).y };
		std::pair<float, float> temp2t = getSeparation(bots[i], temp_t);
		std::pair<float, float> temp2a = getSeparation(bots[i], temp_a);

		wvu_swarm_std_msgs::charger cur_charger;
		cur_charger.target_x = temp2t.first;
		cur_charger.target_y = temp2t.second;
		cur_charger.x = temp2a.first;
		cur_charger.y = temp2a.second;
		cur_charger.occupied = chargers.charger.at(j).occupied;
		_mail.rel_chargerMail.push_back(cur_charger);

#if DEBUG_ChargerMail
		std::cout<<"bot #: "<<i<<" abs_charger: "<<(chargers.charger.at(j).occupied? "true" : "false")<<std::endl;
//		std::cout<<"bot #: "<<i<<" abs_charger: "<<chargers.charger.at(j).x<<","<<chargers.charger.at(j).y<<std::endl;
//		std::cout<<"bot #: "<<i<<" target: "<<temp.first<<","<<temp.second<<std::endl;
//		std::cout<<"bot #: "<<i<<" rel_charger: "<<cur_charger.x<<","<<cur_charger.y<<std::endl;
#endif
	}
#if DEBUG_ChargerMail
	std::cout<<"##############################################"<<std::endl;
#endif
}

//--------------------------------- PUBLIC METHODS ---------------------------------------\\

void Hub::update(wvu_swarm_std_msgs::vicon_bot_array &_b,
wvu_swarm_std_msgs::vicon_points &_t, wvu_swarm_std_msgs::map_levels &_o,
wvu_swarm_std_msgs::flows &_f, wvu_swarm_std_msgs::chargers &_c,
wvu_swarm_std_msgs::energy &_e, wvu_swarm_std_msgs::sensor_data &_sd)
{
	clearHub();
	viconBotArray = _b;
	targets = _t;
	map = _o;
	flows = _f;
	chargers = _c;
	//energy = _e;
	sensor_data = _sd;
	processVicon();//needed cause this data needs to be converted first
	findNeighbors();
}

wvu_swarm_std_msgs::alice_mail_array Hub::getAliceMail() //Gathers all the relative information for a robot into one msg
{
	int cur_sd_index = 0; //fix for sensor data.
	updateSensorDatas();

	wvu_swarm_std_msgs::alice_mail_array to_return;
	for (int i = 0; i < ridOrder.size(); i++)
	{

		wvu_swarm_std_msgs::alice_mail temp;
		addObsMail(i, temp);
		addNeighborMail(i, temp);
		addTargetMail(i, temp);
		addFlowMail(i, temp);
		addContMail(i, temp);
		addChargerMail(i, temp);

		temp.name = ridOrder.at(i);
		temp.sid = bots[i].swarm_id;
		temp.time = bots[i].time;
		temp.x = bots[i].x;
		temp.y = bots[i].y;
		temp.heading = bots[i].heading;
		temp.vision = VISION;

		if (cur_sd_index < sensor_datas.size())
		{
			if (sensor_datas.at(cur_sd_index).rid == ridOrder.at(i))
			{
				temp.sensor_data = sensor_datas.at(cur_sd_index);
				cur_sd_index++;
			}
		}

		to_return.mails.push_back(temp);
	}
	return to_return;
}

void Hub::clearHub() //Clears information about the robots
{
	neighbors.clear();

}
