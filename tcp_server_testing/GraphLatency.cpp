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

#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string.h>
#include <math.h>
#include <unistd.h>

/**
 * Compile
 *
 * g++ GraphLatency.cpp -oGraphLatency.o -lsfml-graphics -lsfml-window -lsfml-system
 *
 *
 *	Graphs from a single column CSV file containing floats
 *	gives the standard deviation and the mean as well as the graph
 */

#define WIDTH 700
#define HEIGHT 500

std::vector<double> data;

sf::VertexArray line;
double max;

int range_max = 200000;

void render(sf::RenderWindow *window)
{
	for (size_t i = 0; i < range_max; i++)
	{
		line[i].position = sf::Vector2f(
				(double) i * (double) WIDTH / (double) range_max,
				(double) HEIGHT - (data[i] * (double) HEIGHT / max));
		line[i].color = sf::Color::White;
	}

	window->draw(line);
}

int main(int argc, char **argv)
{
	const size_t idx = 1;
	std::ifstream fin;
	std::cout << "graphing: " << argv[idx] << std::endl;

	fin.open(argv[idx]);
	if (!fin)
	{
		std::cout << "\033[41;30mCould not open file!!!!!\033[0m" << std::endl;
		return 1;
	}

	max = 0;

	double sum = 0.0;
	std::string dat;
	while (fin >> dat)
	{
		double d_dat = strtod(dat.c_str(), NULL);
		data.push_back(d_dat);
		sum += data.size() < 100000 ? d_dat : 0;

		if (d_dat > max)
			max = d_dat;
	}

	range_max = range_max < data.size() ? range_max : data.size();

	double mean = (sum / (double) range_max);

	std::cout << "Mean : " << mean << std::endl;

	double std_sum = 0;
	for (size_t i = 0; i < range_max; i++)
	{
		std_sum += (data[i] - mean) * (data[i] - mean);
	}
	double stdev = sqrt(1.0 / (range_max - 1) * std_sum);

	std::cout << "STD : " << stdev << std::endl;

	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Latency Plotting",
			sf::Style::Default);

	line = sf::VertexArray(sf::LineStrip, data.size());

	while (window.isOpen()) // main loop
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			switch (event.type)
			// basic event poll
			{
			case sf::Event::Closed: // close window event
				window.close();
				break;
			}
		}

		// drawomg
		if (window.isOpen())
		{
			window.clear();
			render(&window);
			window.display();
		}

		// looking for callbacks
		usleep(1000);
	}

	return 0;
}
