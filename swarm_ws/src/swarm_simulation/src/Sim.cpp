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
#include <swarm_simulation/Flock.h>
#include <swarm_simulation/Body.h>
#include <swarm_simulation/Pvector.h>
#include <swarm_simulation/Sim.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <math.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
#include <swarm_simulation/Font.h>
//#include "ros/ros.h"
bool update;
void Sim::vectorCallback(const wvu_swarm_std_msgs::robot_command_array &msg)
{
	for (int i = 0; i < flock.bodies.size(); i++)
	{

		for (int j = 0; j < msg.commands.size(); j++)
		{
			if (msg.commands.at(j).rid == flock.bodies.at(i).numid)
			{
				/*
				 * Assumptions of simulation drive compared to irl:
				 * Velocity commands sent to servos are linear.
				 * The uncertainty of the drive can be modeled using small amount of noise (not true but oh well ¯\_(ツ)_/¯ ).
				 * No "IMU" is used. Lack of latency should mean that using original command as given should be fine.
				 * (More probably exist but can't think of them.)
				 */

				float speed = 30; //linear velocity of robot, scaled by the simulation units (3 pixels per cm)
				float lw = 25; //max "motor command"
				float kp = 10;
				float theta = msg.commands.at(j).theta / 180 * M_PI; //to radians
				float omega = speed / lw * kp * theta; //again, no kd
				float r = msg.commands.at(j).r;
				if (0 <= theta && theta < M_PI_2 || theta > 3 * M_PI_2)
				{
					if (theta < M_PI)
					{
						flock.bodies.at(i).a = speed - omega;
						if (abs(flock.bodies.at(i).a) > speed)
							flock.bodies.at(i).a =
									flock.bodies.at(i).a > 0.0 ? speed : -speed;

						flock.bodies.at(i).b = speed;
					}
					else
					{
						flock.bodies.at(i).a = speed;
						flock.bodies.at(i).b = speed - omega;
					}
				}
				else if (theta > M_PI && theta <= 3 * M_PI_2)
				{
					flock.bodies.at(i).a = speed;
					flock.bodies.at(i).b = -speed;
				}
				else if (theta <= M_PI && theta >= M_PI_2)
				{
					flock.bodies.at(i).a = -speed;
					flock.bodies.at(i).b = speed;
				}
				else //error case
				{
					flock.bodies.at(i).a = 0;
					flock.bodies.at(i).b = 0;
				}
				if (r < 0.1)
				{
					flock.bodies.at(i).a = 0;
					flock.bodies.at(i).b = 0;
				}

			}
		}

	}
}
void Sim::obsCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
	obstacles = msg;
}

void Sim::chargerCallback(const wvu_swarm_std_msgs::chargers &msg)
{
	chargers = msg;
}

void Sim::targetCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
	targets = msg;
}
void Sim::flowCallback(const wvu_swarm_std_msgs::flows &msg)
{
	flows = msg;
}

void Sim::realBotCallback(const wvu_swarm_std_msgs::vicon_bot_array &robots)
{
	real_bots = robots;
}

// Construct window using SFML
Sim::Sim()
{
	this->bodiesSize = 10.5; //7.5
	sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	this->window_height = O_SIM_HEI;
	this->window_width = O_SIM_WID;
	game = false;
	//std::cout<<"Window h;w = "<<window_height << "; "<<window_width<<"\n";
	this->window.create(
			sf::VideoMode(window_width, window_height, desktop.bitsPerPixel),
			"Swarm Simulation", sf::Style::None);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void Sim::Run(ros::NodeHandle _n)
{
	PrevIteration pI { false, 0, false, false, false, false }; //struct for storing click and drag info.
	std::cout << "initialized pI\n";
	char letters[100] = { 'D', 'E', 'P', 'A', 'N', 'J', 'G', 'A', 'C', 'T', 'M',
			'A', 'M', 'D', 'S', 'C', 'N', 'H', 'V', 'A', 'N', 'Y', 'N', 'C', 'R', 'I',
			'V', 'T', 'K', 'Y', 'T', 'N', 'O', 'H', 'L', 'A', 'I', 'N', 'M', 'S', 'I',
			'L', 'A', 'L', 'M', 'E', 'M', 'O', 'A', 'R', 'M', 'I', 'F', 'L', 'T', 'X',
			'I', 'A', 'W', 'I', 'C', 'A', 'M', 'N', 'O', 'R', 'K', 'A', 'W', 'V', 'N',
			'V', 'N', 'E', 'C', 'O', 'N', 'D', 'S', 'D', 'M', 'T', 'W', 'A', 'I', 'D',
			'W', 'Y', 'U', 'T', 'O', 'K', 'N', 'M', 'A', 'Z', 'A', 'K', 'H', 'I' };

	int x = 50; //x initial positions for the bots.
	for (int i = 0; i < NUMBOTS; i++)
	{
		char temp[2] = { letters[2 * i], letters[2 * i + 1] };
		int y = 50 * (i % 10) + 50; // y pos of bots

//		std::cout<<"----Bot ID: "<<temp[0]<<temp[1]<<"-------\n";
//		std::cout<<"x,y: "<<x<<","<<y<<"\n";

		Body b(x, y, temp, i); // Starts all bodies in the center of the screen
		b.sid = i % 2 + 1;
		sf::CircleShape shape(0);

		// Changing the Visual Properties of the shape.
		shape.setPosition(b.location.x, b.location.y); // Sets position of shape to random location that body was set to.
		shape.setOrigin(10.5, 10.5);
		//shape.setPosition(window_width, window_height); // Testing purposes, starts all shapes in the center of screen.
		if (b.sid == 1)
			shape.setFillColor(sf::Color::Yellow);
		else
			shape.setFillColor(sf::Color::Cyan);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1);
		shape.setRadius(bodiesSize);
		shape.rotate((180.0 / M_PI) * M_PI_2);

		//creates the red line at the head of each bot.
		sf::RectangleShape line(sf::Vector2f(5, 2));
		line.setFillColor(sf::Color::Black);
		line.setPosition(b.location.x, b.location.y);
		line.setOrigin(-2, 1);

		// Adding the body to the flock and adding the shapes to the vector<sf::CircleShape>
		flock.addBody(b);
		shapes.push_back(shape);

		//saves a vector of lines (one for each bot).
		lines.push_back(line);

		//draw all objects on window.
		window.draw(shape);
		window.draw(line);

		if (y == 500) //increments the  x pos so bots are drawn in a grid.
		{
			x += 50;
		}

	}
	window.display();

	//Initializes all publishers and subscribers
	ros::Publisher pub = _n.advertise < wvu_swarm_std_msgs::vicon_bot_array
			> ("vicon_array", 1000); //Publishes like Vicon
	ros::Publisher pub2 = _n.advertise < wvu_swarm_std_msgs::vicon_points
			> ("virtual_targets", 1000);
	ros::Publisher pub3 = _n.advertise < wvu_swarm_std_msgs::vicon_points
			> ("virtual_obstacles", 1000);
	ros::Subscriber sub = _n.subscribe("final_execute", 1000,
			&Sim::vectorCallback, this); //subscribes to funnel
	ros::Subscriber sub2 = _n.subscribe("virtual_obstacles", 1000,
			&Sim::obsCallback, this); //subscribes to virtual obstacles
	ros::Subscriber sub3 = _n.subscribe("virtual_targets", 1000,
			&Sim::targetCallback, this); //gets virtual targets
	ros::Subscriber sub4 = _n.subscribe("virtual_flows", 1000, &Sim::flowCallback,
			this); //gets virtual targets
	ros::Subscriber sub5 = _n.subscribe("chargers", 1000, &Sim::chargerCallback,
			this); //gets virtual targets

	// gets vicon input for mixed sim and real
	ros::Subscriber sub6 = _n.subscribe("real_locations_array", 1000,
			&Sim::realBotCallback, this);
	ros::Rate loopRate(20);

	do //main while loop (runs the simulation).
	{
		pI = HandleInput(pI);
		Render();
		wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages(
				real_bots);
		pub.publish(vb_array); //publishing vicon_bot_array
		pub3.publish(obstacles);
		ros::spinOnce();
		loopRate.sleep();
	} while (window.isOpen() && ros::ok());

}

PrevIteration Sim::HandleInput(PrevIteration _pI)	//handles input to the graphics window
{
	sf::Event event;
	while (window.pollEvent(event))
	{
		int i = 0; //iterator for dragging while loop
		bool found = false; //sentinel for finding a selected bot.
		float mX = event.mouseButton.x; //mouse x pos
		float mY = event.mouseButton.y; //mouse y pos
		pauseSim = false; //pause boolean.

		//---------- Pressing the escape key will close the program
		if ((event.type == sf::Event::Closed)
				|| (event.type == sf::Event::KeyPressed
						&& event.key.code == sf::Keyboard::Escape))
		{
			window.close();
		} //------------------------<swarm_simulation/Sim.h>-----------------------------------

		//------------------allows for pause. Press the Pause button.-----------
		pause(event.type == sf::Event::KeyPressed,
				event.key.code == sf::Keyboard::Space, pauseSim, &window, event);
		clickNdragBots(&_pI, mX, mY, event); //runs click and drag for bots
		clickNdragObstacles(&_pI, mX, mY, event); //allows for click and drag on obstacles
	}
	return _pI; //tracks state of dragging (see sim.h)
}

void Sim::Render() //draws changes in simulation states to the window.
{
	window.clear();
	flock.applyPhysics(&targets);
	drawObstacles();
	drawTargets();
	drawFlows();
	drawChargers();

// Draws all of the bodies out, and applies functions that are needed to update.
	for (int i = 0; i < shapes.size(); i++)
	{ // Matches up the location of the shape to the body
		shapes[i].setPosition(flock.getBody(i).location.x,
				flock.getBody(i).location.y);
		lines[i].setPosition(flock.getBody(i).location.x,
				flock.getBody(i).location.y);

		float theta = 180.0 / M_PI * (flock.bodies.at(i).heading);
		shapes[i].setRotation(90 - theta); //aligns body with direction of motion
		lines[i].setRotation(-theta); //aligns line with direction of motion

		window.draw(shapes[i]);
		window.draw(lines[i]);
	}
	addText();
	window.display(); //updates display
}

void Sim::addText() //adds text for the state abbreviations
{

	sf::Font font;
	font.loadFromMemory(&ComicSansMS3_ttf, ComicSansMS3_ttf_len);

	for (int i = 0; i < shapes.size(); i++)
	{

		//creates text on the bodies

		sf::Text text;
		text.setFont(font);
		text.setCharacterSize(10);
		text.setColor(sf::Color::Red);

		std::string temp(flock.getBody(i).id);
		text.setString(temp.substr(0, 2));
		text.setStyle(sf::Text::Bold);
		text.setOrigin(7.5, 7.5);

		texts.push_back(text);

		window.draw(text);

		window.draw(texts[i]);

		texts[i].setPosition(flock.getBody(i).location.x,
				flock.getBody(i).location.y);
	}
}

void Sim::clickNdragObstacles(PrevIteration *_pI, float _mX, float _mY,
		sf::Event _event) //for click and drag on bots
{
	int i = 0; //iterator for dragging while loop
	bool found = false; //sentinel for finding a selected bot.

	float x_pos;
	float y_pos;

	//----------Allows for click and drag for bots. ------------------------------
	if (_pI->dragging == true && _pI->obs == true)
	{
		obstacles.point.at(_pI->botId).x = (sf::Mouse::getPosition(window).x - 150)
				/ 3;
	}
	if (_event.type == sf::Event::MouseButtonPressed
			&& _event.mouseButton.button == sf::Mouse::Left && _pI->prevClick == true
			&& _pI->obs == true)
	{
		_pI->dragging = false;
		_pI->prevClick = false;
		_pI->obs = false;

	}
	else if (_event.type == sf::Event::MouseButtonPressed
			&& _event.mouseButton.button == sf::Mouse::Left && _pI->prevClick == false
			&& _pI->obs == false && _pI->bot == false && _pI->target == false)
	{
		while (found != true && i < obstacles.point.size())
		{
			x_pos = O_SIM_WID_2 + obstacles.point.at(i).x * 3;
			y_pos = O_SIM_HEI_2 - obstacles.point.at(i).y * 3;
			if (x_pos > _mX - 3 && x_pos < _mX + 3 && y_pos > _mY - 3
					&& y_pos < _mY + 3)
			{
				found = true;
				_pI->botId = i;
				_pI->dragging = true;
				_pI->prevClick = true;
				_pI->obs = true;

			}
			i++;
		}
	}
}

void Sim::clickNdragBots(PrevIteration *_pI, float _mX, float _mY,
		sf::Event _event) //for click and drag on bots
{
	int i = 0; //iterator for dragging while loop
	bool found = false; //sentinel for finding a selected bot.

	//----------Allows for click and drag for bots. ------------------------------
	if (_pI->dragging == true && _pI->bot == true)
	{
		flock.bodies.at(_pI->botId).location.x = sf::Mouse::getPosition(window).x; //event.mouseButton.x;
		flock.bodies.at(_pI->botId).location.y = sf::Mouse::getPosition(window).y; //event.mouseButton.y;
	}
	if (_event.type == sf::Event::MouseButtonPressed
			&& _event.mouseButton.button == sf::Mouse::Left && _pI->prevClick == true
			&& _pI->bot == true)
	{
		_pI->dragging = false;
		_pI->prevClick = false;
		_pI->bot = false;

	}
	else if (_event.type == sf::Event::MouseButtonPressed
			&& _event.mouseButton.button == sf::Mouse::Left && _pI->prevClick == false
			&& _pI->obs == false && _pI->bot == false && _pI->target == false)
	{
		while (found != true)
		{
			if (((flock.bodies.at(i).location.x > _mX - bodiesSize)
					&& (flock.bodies.at(i).location.x < _mX + bodiesSize))
					&& ((flock.bodies.at(i).location.y > _mY - bodiesSize)
							&& (flock.bodies.at(i).location.y < _mY + bodiesSize)))
			{
				found = true;
				_pI->botId = i;
				_pI->dragging = true;
				_pI->prevClick = true;
				_pI->bot = true;
			}
			else if (i == flock.bodies.size() - 1)
			{
				found = true;
			}
			i++;
		}
	}
}

void Sim::clickNdragTarget(PrevIteration *_pI, float _mX, float _mY,
		sf::Event _event) //for click and drag on targets
{
	int i = 0; //iterator for dragging while loop
	bool found = false; //sentinel for finding a selected bot.
	//		----------Allows for click and drag. ------------------------------
	if (_pI->dragging == true && _pI->target == true)
	{
		targets.point.at(i).x = sf::Mouse::getPosition(window).x / 3 - 50;
		; //event.mouseButton.x;
		targets.point.at(i).y = sf::Mouse::getPosition(window).y / -3 + 100; //event.mouseButton.y;
	}
	if (_event.type == sf::Event::MouseButtonPressed
			&& _event.mouseButton.button == sf::Mouse::Left && _pI->prevClick == true
			&& _pI->target == true)
	{
		_pI->dragging = false;
		_pI->prevClick = false;
		_pI->target = false;

	}
	else if (_event.type == sf::Event::MouseButtonPressed
			&& _event.mouseButton.button == sf::Mouse::Left && _pI->prevClick == false
			&& _pI->obs == false && _pI->bot == false && _pI->target == false)
	{
		while (found != true && i < targets.point.size())
		{
			if (((targets.point.at(i).x > _mX / 3 - 50 - 6)
					&& (targets.point.at(i).x < _mX / 3 - 50 + 6))
					&& ((targets.point.at(i).y > _mY / -3 + 100 - 6)
							&& (targets.point.at(i).y < _mY / -3 + 100 + 6)))
			{
				found = true;
				_pI->botId = i;
				_pI->dragging = true;
				_pI->prevClick = true;
				_pI->target = true;
			}
			i++;
		}
	}
}

bool Sim::pause(bool _key_pressed, bool _pause_pressed, bool &_pause_sim,
		sf::RenderWindow *win, sf::Event _event)
{ //checks if pause pressed. Inf loop if so.
	if ((_key_pressed) && (_pause_pressed))
	{
		_pause_sim = true;
		for (int i = 0; i < flock.bodies.size(); i++)
			flock.bodies.at(i).bodyPause = _pause_sim; //need to actually stop the bodies from calculating further positions.
		std::cout << "paused" << std::endl;
	}
	while (_pause_sim == true) //runs while pause in effect.
	{
		if (win->pollEvent(_event))
		{
			if ((_event.type == sf::Event::KeyPressed)
					&& (_event.key.code == sf::Keyboard::Space))
			{ //allows for unpause.
				_pause_sim = false;

				std::cout << "unpaused" << std::endl;
			}
			if ((_event.type == sf::Event::Closed)
					|| (_event.type == sf::Event::KeyPressed
							&& _event.key.code == sf::Keyboard::Escape))
			{
				win->close();
			}
		}
	}
	return _pause_sim;
}

void Sim::drawFlows() //draws flows (basically a vector field)
{
	float x2;
	float y2;

	for (int i = 0; i < flows.flow.size(); i++)
	{
		sf::RectangleShape line(sf::Vector2f(flows.flow.at(i).r, 1));
		line.setPosition(flows.flow.at(i).x * 3 + O_SIM_WID_2,
				O_SIM_HEI_2 - flows.flow.at(i).y * 3);

		x2 = flows.flow.at(i).r * cos(flows.flow.at(i).theta);
		y2 = flows.flow.at(i).r * sin(flows.flow.at(i).theta);

		if (y2 < 0)
		{
			y2 = -y2;
		}
		else
		{
			y2 = -y2;
		}

		//convert to sim frame
		line.setFillColor(sf::Color::White);
		line.setOrigin(0, 0);
		line.setRotation(-flows.flow.at(i).theta * 180 / M_PI);
		line.setOutlineColor(sf::Color::Black);
		line.setOutlineThickness(1);
		window.draw(line);

		sf::RectangleShape line1(sf::Vector2f(1, 1)); //gives vector direction
		line1.setFillColor(sf::Color::Black);
		line1.setPosition(flows.flow.at(i).x * 3 + O_SIM_WID_2 + x2,
				O_SIM_HEI_2 + y2 - flows.flow.at(i).y * 3);
		line1.setOrigin(-1, 0);
		line1.setRotation(-flows.flow.at(i).theta * 180 / M_PI + 130);
		line1.setOutlineColor(sf::Color::Black);
		line1.setOutlineThickness(1);
		window.draw(line1);
	}
}

//------------------------Drawing Code --------------------------------------------
#define DRAW(list, cont) \
	for (int i = 0;i < list.point.size();i++)\
	{\
		sf::CircleShape shape(0);\
		shape.setPosition(targets.point.at(i).x * 3 + O_SIM_WID_2,\
							O_SIM_HEI_2 - targets.point.at(i).y * 3);\
		cont;\
		window.draw(shape);\
	}\

void Sim::drawTargets() //draws targets
{
	DRAW(targets,
	{
		shape.setOrigin(10, 10);
		shape.setFillColor(sf::Color::Green);
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(1);
		shape.setRadius(10);
	});
}

void Sim::drawObstacles() //draws obstacles
{
	DRAW(obstacles,
	{
		shape.setOrigin(2, 2);
		shape.setFillColor(sf::Color::Blue);
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(1);
		shape.setRadius(3);
	});
}

void Sim::drawChargers() //draws chargers
{
	float x_adjust = 0;
	float y_adjust;
	for (int i = 0; i < chargers.charger.size(); i++)
	{
		sf::RectangleShape shape(sf::Vector2f(5, 12)); //gives vector direction
		if (chargers.charger.at(i).occupied == true)
		{
			shape.setFillColor(sf::Color::Green);
		}
		else
		{
			shape.setFillColor(sf::Color::Red);
		}
		if (chargers.charger.at(i).x > 0)
		{
			x_adjust = -x_adjust;
		}
		if (chargers.charger.at(i).y < 0)
		{
			y_adjust = -y_adjust;
		}
		shape.setPosition(chargers.charger.at(i).x * 3 + O_SIM_WID_2 + x_adjust,
				O_SIM_HEI_2 - chargers.charger.at(i).y * 3);
		shape.setOrigin(0, 6);
		window.draw(shape);
	}
}

//-----------------------------------------------------------------------------------
