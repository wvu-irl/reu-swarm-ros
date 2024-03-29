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

#include <iostream>
#include "Flock.h"
#include "Body.h"
#include "Pvector.h"
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
#include <wvu_swarm_std_msgs/flows.h>
#include <wvu_swarm_std_msgs/virtual_objects.h>
#include <wvu_swarm_std_msgs/chargers.h>

#include "sim_settings.h"
//#include <swarm_server/robot_id.h>

#ifndef SIM_H
#define SIM_H

// Sim handles the instantiation of a flock of bodies, Sim input, asks the
// model to compute the next step in the stimulation, and handles all of the
// program's interaction with SFML.

struct PrevIteration //struct created solely for click and drag of bots, targets, and obstacles.
{
	bool dragging;
	int botId;
	bool prevClick;
	bool bot; //is true if bot selected.
	bool target; //is true if target selected.
	bool obs; //is true if obstacle selected.
};

class Sim
{

private:
	sf::RenderWindow window;
	int window_width;
	int window_height;
	bool pauseSim;
	Flock flock;
	vector<sf::CircleShape> shapes;
	vector<sf::RectangleShape> lines;
	vector<sf::Text> texts;
	//vector<wvu_swarm_std_msgs::vicon_point> obstacles;
	wvu_swarm_std_msgs::vicon_bot_array real_bots;
	wvu_swarm_std_msgs::vicon_points obstacles;
	wvu_swarm_std_msgs::vicon_points targets;
	wvu_swarm_std_msgs::chargers chargers;
	wvu_swarm_std_msgs::flows flows;
	vector<sf::CircleShape> obs_shapes;
	bool game;
	string winner;


	//text related stuff
	void addText();
	float bodiesSize;

	//Handle graphics rendering and mouse/key inputs
	PrevIteration HandleInput(PrevIteration _pI);
	void Render();
	void clickNdragBots(PrevIteration *_pI, float _mX, float _mY, sf::Event _event);
	void clickNdragTarget(PrevIteration *_pI,float _mX, float _mY, sf::Event _event);
	void clickNdragObstacles(PrevIteration *_pI, float _mX, float _mY, sf::Event _event);
	bool pause(bool _key_pressed, bool _pause_pressed, bool &_pause_sim, sf::RenderWindow* win, sf::Event _event);

	//subscriber input handleing
	void vectorCallback(const wvu_swarm_std_msgs::robot_command_array &msg); //processes info from final_execute
	void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg);   //processes info from virtual_obstacles
	void targetCallback(const wvu_swarm_std_msgs::vicon_points &msg);   //processes info from virtual_targets
	void flowCallback(const wvu_swarm_std_msgs::flows &msg);         //processes info from virtual_flows
	void chargerCallback(const wvu_swarm_std_msgs::chargers &msg);   //processes info from chargers
	void realBotCallback(const wvu_swarm_std_msgs::vicon_bot_array &robots); // processes info about real robots so that sim can mix

	//Specific objects to be rendered.
	void drawObstacles();
	void drawTargets();
	void updateTargetPos();
	void drawFlows();
	void drawChargers(); //draws chargers

public:
	Sim();
	void Run(ros::NodeHandle _n);
};

#endif
