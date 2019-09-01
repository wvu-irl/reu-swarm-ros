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

/**
 * 
 * g++ calibrate.cpp -o Calibrate.o -lsfml-graphics -lsfml-window -lsfml-system
 * 
 */
#include <ros/ros.h>
#include <SFML/Graphics.hpp>
#include <unistd.h>
#include <math.h>
#include <vector>

#include <iostream>
#include <fstream>

#include <visualization/visualization_settings.h>
// definition for debugging messages
#define DEBUG 0

// dot product operator
double operator*(sf::Vector2f a, sf::Vector2f b)
{
	return a.x * b.x + a.y * b.y;
}

// instriction list
const sf::String instructions[] = { "Select TOP LEFT", "Select TOP RIGHT",
		"Select BOTTOM RIGHT", "Select BOTTOM LEFT",
		"Press space to write or R to reset" };

// current instruction
static int state = 0;

// calibration values struct
typedef struct
{
	sf::Vector2f top_left;
	sf::Vector2f top_right;
	sf::Vector2f bottom_right;
	sf::Vector2f bottom_left;
} calibrate;

// drawing instructions
sf::Font fon;
sf::Text instruct;

// current calibration values
calibrate g_calib;

// window scale
// if the window changes size SFML changes scale
double scaleX = 1, scaleY = 1;

/**
 * Renders all the necessary visual elements to the window
 *
 * window is a pointer to the render window created in main
 */
void render(sf::RenderWindow *window)
{
#if DEBUG
	std::cout << "Started Rendering" << std::endl;
#endif
	// creating vertex array to draw later
	sf::VertexArray trap(sf::LinesStrip, state + 1 + (state == 3 ? 1 : 0));
#if DEBUG
	std::cout << "Created vertex array size : "
			<< (state + 1 + (state == 3 ? 1 : 0)) << std::endl;
#endif
	switch (state) // checking state
	{
	default:
	case 3: // cascading switch for each point
#if DEBUG
		std::cout << "Drawing state 3" << std::endl;
#endif
		trap[4].position = g_calib.top_left;
		trap[4].color = sf::Color::White;
		trap[3].position = g_calib.bottom_left;
		trap[3].color = sf::Color::White;
	case 2:
#if DEBUG
		std::cout << "Drawing state 2" << std::endl;
#endif
		trap[2].position = g_calib.bottom_right;
		trap[2].color = sf::Color::White;
	case 1:
#if DEBUG
		std::cout << "Drawing state 1" << std::endl;
#endif
		trap[1].position = g_calib.top_right;
		trap[1].color = sf::Color::White;
	case 0:
#if DEBUG
		std::cout << "Drawing state 0" << std::endl;
#endif
		trap[0].position = g_calib.top_left;
		trap[0].color = sf::Color::White;
	}
#if DEBUG
	std::cout << "Drawing lines" << std::endl;
#endif
	window->draw(trap); // draw vertex array
	window->draw(instruct); // draw instruction text
}

// sets the text of the instruction and re-centers it
void changeInstruction(sf::String str)
{
	instruct.setString(str);
	instruct.setPosition(
			sf::Vector2f(WIDTH / 2 - instruct.getLocalBounds().width / 2,
			HEIGHT / 2));
}

// main for hte node
int main(int argc, char **argv)
{
	ros::init(argc, argv, "calibrate");

#if DEBUG
	std::cout << "Started" << std::endl;
#endif
	// initial calibration state
	state = 0;
	g_calib.top_left = sf::Vector2f(0, 0);
	g_calib.top_right = sf::Vector2f(0, 0);
	g_calib.bottom_right = sf::Vector2f(0, 0);
	g_calib.bottom_left = sf::Vector2f(0, 0);

	// loading font
	fon.loadFromFile(sf::String("src/visualization/assets/ComicSansMS3.ttf"));
	// setting up instruction text
	instruct.setColor(sf::Color::White);
	instruct.setFont(fon);
	instruct.setCharacterSize(20);
	changeInstruction(instructions[0]);
#if DEBUG
	std::cout << "Loaded instruction" << std::endl;
#endif
	// creating render window
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Projector Calibration",
			sf::Style::Default);
#if DEBUG
	std::cout << "Loaded window" << std::endl;
#endif
	while (window.isOpen()) // main loop
	{
#if DEBUG
		std::cout << "Entered loop" << std::endl;
#endif
		sf::Event event;
		while (window.pollEvent(event))
		{
#if DEBUG
			std::cout << "Entered event loop" << std::endl;
#endif
			switch (event.type) // event handling
			{
			case sf::Event::Closed: // close window event
#if DEBUG
				std::cout << "Close Event" << std::endl;
#endif
				window.close();
				break;
			case sf::Event::MouseMoved: // mouse moved event
#if DEBUG
				std::cout << "Mouse moved Event" << std::endl;
#endif
				switch (state) // checking which vertex to move
				{
				case 0:
					// each mouse move is scaled by the current scale
					g_calib.top_left = sf::Vector2f(event.mouseMove.x * scaleX,
							event.mouseMove.y * scaleY);
					break;
				case 1:
					g_calib.top_right = sf::Vector2f(event.mouseMove.x * scaleX,
							event.mouseMove.y * scaleY);
					break;
				case 2:
					g_calib.bottom_right = sf::Vector2f(event.mouseMove.x * scaleX,
							event.mouseMove.y * scaleY);
					break;
				case 3:
					g_calib.bottom_left = sf::Vector2f(event.mouseMove.x * scaleX,
							event.mouseMove.y * scaleY);
					break;
				default:
					break;
				}
				break;
			case sf::Event::Resized: // window resize event
				// recalculating scale factors
				scaleX = (float) WIDTH / (float) event.size.width;
				scaleY = (float) HEIGHT / (float) event.size.height;
				break;
			case sf::Event::MouseButtonReleased: // mouse pressed
#if DEBUG
				std::cout << "Mouse released" << std::endl;
#endif
				state += state < 4 ? 1 : 0; // incrementing state reasonably
				changeInstruction(instructions[state]); // updating instructions
				break;
			case sf::Event::KeyPressed: // keyboard events
#if DEBUG
				std::cout << "Key Press event" << std::endl;
#endif
				if (event.key.code == sf::Keyboard::R) // reset key
				{
					state = 0;
					changeInstruction(sf::String("Select TOP LEFT"));
				}
				else if (state == 4 && event.key.code == sf::Keyboard::Space) // write event
				{
					// writing vectors to csv file
					std::ofstream file;
					file.open("src/visualization/cfg/calib.config", std::ios::out); // file location
					if (file) // checking for errors
					{
						// writing values
						file << g_calib.top_left.x << "," << g_calib.top_left.y
								<< std::endl;
						file << g_calib.top_right.x << "," << g_calib.top_right.y
								<< std::endl;
						file << g_calib.bottom_right.x << "," << g_calib.bottom_right.y
								<< std::endl;
						file << g_calib.bottom_left.x << "," << g_calib.bottom_left.y
								<< std::endl;
						file.close();
					}
					else
					{
						std::cout << "\033[30;41mDid not write\033[0m" << std::endl;
					}
					changeInstruction("Done, press R to re-calibrate"); // final instruction
				}
				break;
			}
		}

#if DEBUG
		std::cout << "rendering" << std::endl;
#endif
		if (window.isOpen())
		{
			window.clear();
			render(&window);
			window.display();
		}
#if DEBUG
		std::cout << "Sleeping" << std::endl;
#endif
		usleep(10000);
	}

	return 0;
}
