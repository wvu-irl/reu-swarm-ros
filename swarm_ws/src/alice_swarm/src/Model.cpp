#include "alice_swarm/Model.h"
#include "alice_swarm/aliceStructs.h"
#include "wvu_swarm_std_msgs/alice_mail_array.h"
#include "wvu_swarm_std_msgs/map.h"
#include "wvu_swarm_std_msgs/gaussian.h"
#include "wvu_swarm_std_msgs/neighbor_mail.h"
#include <iostream>

Model::Model()
{
	first = true;
	name = 0;
}

Model::Model(int _name)
{
	first = true;
	name = _name;
}

void Model::clear()
{
	obstacles.clear();
	flows.clear();
	neighbors.clear();
	targets.clear();
}

void Model::archiveAdd(AliceStructs::mail &_toAdd)
{
	float vision = _toAdd.vision - 0; // make number larger if we want to account for moving things, will be buggy tho

	//placing objects that exited fov
	for (auto& obstacle : obstacles)
	{
		if (pow(pow(obstacle.x_off - _toAdd.xpos, 2) + pow(obstacle.y_off - _toAdd.ypos, 2), 0.5) > vision)
		{
			obstacle.time = time;
			obstacle.observer_name = name;
			archived_obstacles.push_back(obstacle);
		}

	}
	for (auto& tar : targets)
	{
		if (pow(pow(tar.x - _toAdd.xpos, 2) + pow(tar.y - _toAdd.ypos, 2), 0.5) > vision)
		{
			tar.time = time;
			tar.observer_name = name;
			archived_targets.push_back(tar);
		}
	}

	if (time.nsec % 10 == 0) //so we don't record literally every time step
	{
		AliceStructs::pose temp(cur_pose);
		temp.observer_name = name;
		temp.time = time;
		archived_contour.push_back(temp);
	}
}

void Model::sensorUpdate(AliceStructs::mail &_toAdd)
{
	clear();
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
	for (auto& obstacle : _toAdd.obstacles)
	{
		obstacles.push_back(obstacle);
	}
	for (auto& flow : _toAdd.flows)
	{
		flows.push_back(flow);
	}
	for (auto& neighbor : _toAdd.neighbors)
	{
		neighbors.push_back(neighbor);
	}
	for (auto& tar : _toAdd.targets)
	{
		targets.push_back(tar);
	}
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
	for (auto& obstacle : obstacles)
	{
		wvu_swarm_std_msgs::obs_msg obs_msg;
		obs_msg.ellipse.x_rad = obstacle.x_rad;
		obs_msg.ellipse.y_rad = obstacle.y_rad;
		obs_msg.ellipse.theta_offset = obstacle.theta_offset - (cur_pose.heading - first_pose.heading);
		obs_msg.ellipse.offset_x = obstacle.x_off + (cur_pose.x - first_pose.x);
		obs_msg.ellipse.offset_y = obstacle.y_off + (cur_pose.y - first_pose.y);
		obs_msg.time = time;
		obs_msg.observer = name;
		map.obsMsg.push_back(obs_msg);
	}

	//inserts obstacles from its extended knowledge
	for (auto& obstacle : archived_obstacles)
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
	for (auto& tar : targets)
	{
		wvu_swarm_std_msgs::tar_msg tar_msg;
		tar_msg.pointMail.x = tar.x;
		tar_msg.pointMail.y = tar.y;
		tar_msg.time = time;
		tar_msg.observer = name;
		map.tarMsg.push_back(tar_msg);
	}

	//inserts targets from its extended knowledge
	for (auto& tar : archived_targets)
	{
		wvu_swarm_std_msgs::tar_msg tar_msg;
		tar_msg.pointMail.x = tar.x;
		tar_msg.pointMail.y = tar.y;
		tar_msg.time = tar.time;
		tar_msg.observer = tar.observer_name;
		map.tarMsg.push_back(tar_msg);
	}

	//inserts targets from its own vision
	for (auto& tar : targets)
	{
		wvu_swarm_std_msgs::tar_msg tar_msg;
		tar_msg.pointMail.x = tar.x;
		tar_msg.pointMail.y = tar.y;
		tar_msg.time = time;
		tar_msg.observer = name;
		map.tarMsg.push_back(tar_msg);
	}

	//inserts contour values from its knowledge
	for (auto& cont : archived_contour)
	{
		wvu_swarm_std_msgs::cont_msg cont_msg;
		cont_msg.pointMail.x = cont.x;
		cont_msg.pointMail.y = cont.y;
		cont_msg.contVal = cont.z;
		cont_msg.time = time;
		cont_msg.observer = cont.observer_name;
		map.contMsg.push_back(cont_msg);
	}

	_pub.publish(map);
}

void Model::forget()
{
//	for (std::vector<AliceStructs::obj>::iterator it = archived_obstacles.begin(); it != archived_obstacles.end(); ++it)
//	{
//		if (it->time.sec - time.sec > 10)
//			archived_obstacles.erase(it);
//	}
//	for (std::vector<AliceStructs::pnt>::iterator it = archived_targets.begin(); it != archived_targets.end(); ++it)
//	{
//		if (it->time.sec - time.sec > 10)
//			archived_targets.erase(it);
//	}
	std::vector<AliceStructs::pose>::iterator it = archived_contour.begin();
	while (it != archived_contour.end())
	{
		if (time.sec - it->time.sec > 10)
		{
			std::cout << time.sec - it->time.sec << std::endl;
			it = archived_contour.erase(it);
		}
		else it++;
	}
//	std::cout << "oy" << std::endl;
}
