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
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/map.h>
#include <swarm_simulation/Font.h>
#ifndef POV_H
#define POV_H

class AlicePOV
{
private:
	sf::RenderWindow window;
	sf::RenderWindow window2;
	int window_width;
	int window_height;

	wvu_swarm_std_msgs::alice_mail_array mail;
	wvu_swarm_std_msgs::map map;
	int name;
//	vector<sf::CircleShape> neighbors;
//	vector<sf::CircleShape> neighbors;
//	vector<sf::RectangleShape> lines; //for flows
//	vector<sf::Text> texts;
//	vector<sf::CircleShape> obs_shapes; //for new obstacles

//text related stuff
//void addText();
	float bodiesSize;

	//subscriber input handleing
	void mailCallback(const wvu_swarm_std_msgs::alice_mail_array &msg);
	void mapCallback(const wvu_swarm_std_msgs::map &msg);
	void HandleInput();
	void drawMail(ros::ServiceClient _client);
	void drawMsg(ros::ServiceClient _client);
public:
	AlicePOV();
	void Run(ros::NodeHandle _n);
	void Render(ros::ServiceClient _client);

};

#endif
