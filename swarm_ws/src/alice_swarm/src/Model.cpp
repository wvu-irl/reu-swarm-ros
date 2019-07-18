#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/map.h"
#include "wvu_swarm_std_msgs/gaussian.h"
#include <alice_swarm/get_maps.h>
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include <iostream>
#include <chrono>
Model::Model()
{
	first = true;
	name = 0;
}

Model::Model(int _name)
{
	first = true;
	name = _name;
	committed = false;
}

void Model::clear()
{
	obstacles.clear();
	flows.clear();
	neighbors.clear();
	targets.clear();
	rel_chargers.clear();
}
std::pair<float, float> Model::transformCur(float _x, float _y) //goes from cur frame to first frame.
{
	std::pair<float, float> to_return;
	float abs_x = _x * cos(cur_pose.heading) + _y * -sin(cur_pose.heading);
	float abs_y = _x * sin(cur_pose.heading) + _y * cos(cur_pose.heading);
	abs_x = abs_x + cur_pose.x - first_pose.x;
	abs_y = abs_y + cur_pose.y - first_pose.y;
	to_return.first = abs_x * cos(-first_pose.heading) + abs_y * -sin(-first_pose.heading);
	to_return.second = abs_x * sin(-first_pose.heading) + abs_y * cos(-first_pose.heading);
	return to_return;
}

std::pair<float, float> Model::transformFir(float _x, float _y) //goes from first frame to cur frame
{
	std::pair<float, float> to_return;
	float abs_x = _x * cos(first_pose.heading) + _y * -sin(first_pose.heading);
	float abs_y = _x * sin(first_pose.heading) + _y * cos(first_pose.heading);
	abs_x = abs_x - cur_pose.x + first_pose.x;
	abs_y = abs_y - cur_pose.y + first_pose.y;
	to_return.first = abs_x * cos(-cur_pose.heading) + abs_y * -sin(-cur_pose.heading);
	to_return.second = abs_x * sin(-cur_pose.heading) + abs_y * cos(-cur_pose.heading);
	return to_return;
}
std::pair<float, float> Model::transformFtF(float _x, float _y, float _ox, float _oy, float _oheading)
{ //convert specified frame to first frame.
	std::pair<float, float> to_return;
	float abs_x = _x * cos(_oheading) + _y * -sin(_oheading);
	float abs_y = _x * sin(_oheading) + _y * cos(_oheading);
	abs_x = abs_x - first_pose.x + _ox;
	abs_y = abs_y - first_pose.y + _oy;
	to_return.first = abs_x * cos(-first_pose.heading) + abs_y * -sin(-first_pose.heading);
	to_return.second = abs_x * sin(-first_pose.heading) + abs_y * cos(-first_pose.heading);
	return to_return;
}

void Model::archiveAdd(AliceStructs::mail &_toAdd)
{
	vision = _toAdd.vision - 0; // make number larger if we want to account for moving things, will be buggy tho
	//placing objects that exited fov
	for (auto &obstacle : obstacles)
	{
		if (pow(pow(obstacle.x_off - _toAdd.xpos, 2) + pow(obstacle.y_off - _toAdd.ypos, 2), 0.5) > vision)
		{
			std::pair<float, float> temp = transformCur(obstacle.x_off, obstacle.y_off);
			obstacle.x_off = temp.first;
			obstacle.y_off = temp.second;
			obstacle.time = time;
			obstacle.observer_name = name;
			archived_obstacles.push_back(obstacle);
		}

	}
	for (auto &tar : targets)
	{
		if (pow(pow(tar.x - _toAdd.xpos, 2) + pow(tar.y - _toAdd.ypos, 2), 0.5) > vision)
		{
			tar.time = time;
			tar.observer_name = name;
			std::pair<float, float> temp = transformCur(tar.x, tar.y);
			tar.x = temp.first;
			tar.y = temp.second;
			archived_targets.push_back(tar);
		}
	}

//	if (time.nsec % 10 == 0) //so we don't record literally every time step
	//{
	AliceStructs::pose temp(cur_pose);
	std::pair<float, float> temp2 = transformCur(0, 0);
	temp.x = temp2.first;
	temp.y = temp2.second;
	temp.observer_name = name;
	temp.time = time;
	archived_contour.push_back(temp);
//	}
}

void Model::sensorUpdate(AliceStructs::mail &_toAdd)
{
	clear();
	battery_lvl = _toAdd.battery_lvl; //updates battery level.
	battery_state = _toAdd.battery_state; //updates battery state.
	energy = _toAdd.energy; //updates energy
	if (first)
	{
		first_pose.x = _toAdd.xpos;
		first_pose.y = _toAdd.ypos;
		first_pose.heading = _toAdd.heading;
		first = false;
	}
	cur_pose.x = _toAdd.xpos;
	cur_pose.y = _toAdd.ypos;
	cur_pose.z = _toAdd.contVal;
	cur_pose.heading = _toAdd.heading;
	name = _toAdd.name;
	time = _toAdd.time;

	for (auto &obstacle : _toAdd.obstacles)
	{
		obstacles.push_back(obstacle);
	}
	for (auto &flow : _toAdd.flows)
	{
		flows.push_back(flow);
	}
	for (auto &neighbor : _toAdd.neighbors)
	{
		neighbors.push_back(neighbor);
	}
	for (auto &tar : _toAdd.targets)
	{
		targets.push_back(tar);
	}
	//Im pretty sure this does the same as the above loops.
//	obstacles =  _toAdd.obstacles;
//	flows =  _toAdd.flows;
//	neighbors =  _toAdd.neighbors;
//	targets =  _toAdd.targets;

	abs_chargers = _toAdd.abs_chargers; //pointer to absolute chargers vector
	rel_chargers = _toAdd.rel_chargers; //copy of chargers vector with pos in current frame, not absolute.
	priority = _toAdd.priority;

//	int i=0;
//	for (auto& charger : _toAdd.chargers)
//	{
//		chargers.push_back(charger);
//		std::cout<<"==============================\n";
//		std::cout<<"x,y: "<<chargers.at(i).x<<", "<<chargers.at(i).y<<std::endl;
//		i++;
//	}
//	std::cout<<"==============================\n";

// debugging print statments for chargers data transfer;
//	for(int i = 0; i < chargers->size(); i++)
//	{
//		std::cout<<"==============================\n";
//		std::cout<<"x,y: "<<chargers->at(i).x<<", "<<chargers->at(i).y<<" | "<<(chargers->at(i).occupied ? "true" : "false")<<std::endl;
//		std::cout<<"x,y: "<<chargers->at(i).x<<", "<<chargers->at(i).y<<" | "<<(chargers->at(i).occupied ? "true" : "false")<<std::endl;
//	}
//	std::cout<<"==============================\n";
}

void Model::pass(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::map map;
	map.name = name;
	map.x = cur_pose.x;
	map.y = cur_pose.y;
	map.heading = cur_pose.heading;
	map.ox = first_pose.x;
	map.oy = first_pose.y;
	map.oheading = first_pose.heading;

	//inserts obstacles from its own vision
	for (auto &obstacle : obstacles)
	{
		wvu_swarm_std_msgs::obs_msg obs_msg;
		obs_msg.ellipse.x_rad = obstacle.x_rad;
		obs_msg.ellipse.y_rad = obstacle.y_rad;
		obs_msg.ellipse.theta_offset = obstacle.theta_offset - (cur_pose.heading - first_pose.heading);
		std::pair<float, float> temp = transformCur(obstacle.x_off, obstacle.y_off);
		obs_msg.ellipse.offset_x = temp.first;
		obs_msg.ellipse.offset_y = temp.second;
		obs_msg.time = time;
		obs_msg.observer = name;
		map.obsMsg.push_back(obs_msg);
	}

	//inserts obstacles from its extended knowledge
	for (auto &obstacle : archived_obstacles)
	{
		wvu_swarm_std_msgs::obs_msg obs_msg;
		obs_msg.ellipse.x_rad = obstacle.x_rad;
		obs_msg.ellipse.y_rad = obstacle.y_rad;
		obs_msg.ellipse.theta_offset = obstacle.theta_offset;
		obs_msg.ellipse.offset_x = obstacle.x_off;
		obs_msg.ellipse.offset_y = obstacle.y_off;
		obs_msg.time = obstacle.time;
		obs_msg.observer = obstacle.observer_name;
		map.obsMsg.push_back(obs_msg);
	}

	//inserts targets from its own vision
	for (auto &tar : targets)
	{
		wvu_swarm_std_msgs::tar_msg tar_msg;
		std::pair<float, float> temp = transformCur(tar.x, tar.y);
		tar_msg.pointMail.x = temp.first;
		tar_msg.pointMail.y = temp.second;
		tar_msg.time = time;
		tar_msg.observer = name;
		map.tarMsg.push_back(tar_msg);
	}

	//inserts targets from its extended knowledge
	for (auto &tar : archived_targets)
	{
		wvu_swarm_std_msgs::tar_msg tar_msg;
		tar_msg.pointMail.x = tar.x;
		tar_msg.pointMail.y = tar.y;
		tar_msg.time = tar.time;
		tar_msg.observer = tar.observer_name;
		map.tarMsg.push_back(tar_msg);
	}

	//inserts contour values from its knowledge
	for (auto &cont : archived_contour)
	{
		wvu_swarm_std_msgs::cont_msg cont_msg;
		cont_msg.pointMail.x = cont.x;
		cont_msg.pointMail.y = cont.y;
		cont_msg.contVal = cont.z;
		cont_msg.time = cont.time;
		cont_msg.observer = cont.observer_name;
		map.contMsg.push_back(cont_msg);
	}

	//passes the goTo point from the first frame
//	std::pair<float, float> temp = transformCur(goTo.x, goTo.y);
//	map.goToX = temp.first;
//	map.goToY = temp.second;
	map.goToX=goTo.x;
	map.goToY=goTo.y;
	_pub.publish(map);
}

void Model::receiveMap(std::vector<wvu_swarm_std_msgs::map> &_maps, std::vector<int> &_ids)
{
	auto start = std::chrono::high_resolution_clock::now(); //timer for measuring the runtime of map

	for (int i = 0; i < neighbors.size(); i++) //iterate through neighbors
	{
		for (int j = 0; j < _ids.size(); j++) //through id's
		{
			if (_ids.at(j) == neighbors.at(i).name) //to find matching id's
			{
				for (int k = 0; k < _maps.at(j).obsMsg.size(); k++) //copy over all of the obstacles
				{
					wvu_swarm_std_msgs::obs_msg temp = _maps.at(j).obsMsg.at(k);
					if (temp.observer != name) //as long as the observer was someone else
					{
						AliceStructs::obj obstacle;
						std::pair<float, float> temp2 = transformFtF(temp.ellipse.offset_x, temp.ellipse.offset_y, _maps.at(j).ox,
								_maps.at(j).oy, _maps.at(j).oheading);
						obstacle.x_off = temp2.first;
						obstacle.y_off = temp2.second;
						obstacle.x_rad = temp.ellipse.x_rad;
						obstacle.y_rad = temp.ellipse.y_rad;
						obstacle.theta_offset = temp.ellipse.theta_offset;
						obstacle.time = temp.time;
						obstacle.observer_name = temp.observer;
						archived_obstacles.push_back(obstacle);
					}
				}
				for (int k = 0; k < _maps.at(j).tarMsg.size(); k++) //copy over all of the targets
				{
					wvu_swarm_std_msgs::tar_msg temp = _maps.at(j).tarMsg.at(k);
					if (temp.observer != name) //as long as the observer was someone else
					{
						AliceStructs::pnt target;
						std::pair<float, float> temp2 = transformFtF(temp.pointMail.x, temp.pointMail.y, _maps.at(j).ox,
								_maps.at(j).oy, _maps.at(j).oheading);
						target.x = temp2.first;
						target.y = temp2.second;
						target.time = temp.time;
						target.observer_name = temp.observer;
						archived_targets.push_back(target);
					}
				}
				for (int k = 0; k < _maps.at(j).contMsg.size(); k++) //copy over known contour points
				{
					wvu_swarm_std_msgs::cont_msg temp = _maps.at(j).contMsg.at(k);
					if (temp.observer != name) //as long as the observer was someone else
					{
						AliceStructs::pose cont;
						std::pair<float, float> temp2 = transformFtF(temp.pointMail.x, temp.pointMail.y, _maps.at(j).ox,
								_maps.at(j).oy, _maps.at(j).oheading);
						cont.x = temp2.first;
						cont.y = temp2.second;
						cont.z = temp.contVal;
						cont.time = temp.time;
						cont.observer_name = temp.observer;
						archived_contour.push_back(cont);
					}
				}
			}
		}

	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (stop - start);
	//std::cout << "Time taken by srv: " << duration.count() << " microseconds" << std::endl;
}

void Model::forget()
{
	float TOLERANCE = 5; //objects within this distance in x and y will be simplified to one object
	forgetObs(TOLERANCE);
	forgetTargets(TOLERANCE);
	forgetContour(TOLERANCE);
}

void Model::forgetContour(int TOLERANCE) //erases Contours based on time stamp or duplicity.
{
	std::vector<AliceStructs::pose>::iterator it = archived_contour.begin();
	while (it != archived_contour.end())
	{
		if (time.sec - it->time.sec > 10) //if the data is old, delete it
		{
			it = archived_contour.erase(it);
		} else
		{
			std::vector<AliceStructs::pose>::iterator iit = it; //second iterator for comparisons
			iit++;
			while (iit != archived_contour.end()) //checks for repeat elements within the given tolerance
			{
				if (abs(iit->x - it->x) < TOLERANCE && abs(iit->y - it->y) < TOLERANCE)
				{
					if (time.sec - it->time.sec > 1 && iit->time > it->time)
						std::swap(*iit, *it); //takes the newer data
					iit = archived_contour.erase(iit);

				} else
					iit++;
			}
			it++;
		}
	}
}

void Model::forgetTargets(int TOLERANCE) //erases targets based on time stamp or duplicity.
{
	std::vector<AliceStructs::pnt>::iterator it = archived_targets.begin();
	while (it != archived_targets.end())
	{
		std::pair<float, float> temp = transformFir(it->x, it->y);

		if (time.sec - it->time.sec > 10) //if the data is old, delete it
		{
			it = archived_targets.erase(it);
		} else if (pow(pow(temp.first, 2) + pow(temp.second, 2), 0.5) < vision) //if the object should be visible from current position but isn't, delete it
		{
			it = archived_targets.erase(it);
		} else
		{
			std::vector<AliceStructs::pnt>::iterator iit = it; //second iterator for comparisons
			iit++;
			while (iit != archived_targets.end()) //checks for repeat elements within the given tolerance
			{
				if (abs(iit->x - it->x) < TOLERANCE && abs(iit->y - it->y) < TOLERANCE)
				{
					if (time.sec - it->time.sec > 1 && iit->time > it->time)
						std::swap(*iit, *it); //takes the newer data
					iit = archived_targets.erase(iit);

				} else
					iit++;
			}
			it++;
		}
	}
}

void Model::forgetObs(int TOLERANCE) //erases obstacles based on time stamp or duplicity.
{
	std::vector<AliceStructs::obj>::iterator it = archived_obstacles.begin();
	while (it != archived_obstacles.end())
	{
		std::pair<float, float> temp = transformFir(it->x_off, it->y_off);
		if (time.sec - it->time.sec > 10) //if the data is old, delete it
		{
			it = archived_obstacles.erase(it);
		} else if (pow(pow(temp.first, 2) + pow(temp.second, 2), 0.5) < vision) //if the object should be visible from current position but isn't, delete it
		{
			it = archived_obstacles.erase(it);
		} else
		{
			std::vector<AliceStructs::obj>::iterator iit = it; //second iterator for comparisons
			iit++;
			while (iit != archived_obstacles.end()) //checks for repeat elements within the given tolerance
			{
				if (abs(iit->x_off - it->x_off) < TOLERANCE && abs(iit->y_off - it->y_off) < TOLERANCE)
				{
					if (time.sec - it->time.sec > 1 && iit->time > it->time)
						std::swap(*iit, *it); //takes the newer data
					iit = archived_obstacles.erase(iit);

				} else
					iit++;
			}
			it++;
		}
	}
}
