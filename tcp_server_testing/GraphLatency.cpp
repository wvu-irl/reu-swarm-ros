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

int range_max = 100000;

void render(sf::RenderWindow *window)
{
	for (size_t i = 0; i < range_max; i++)
	{
		line[i].position = sf::Vector2f((double)i * (double)WIDTH / (double)range_max, (double)HEIGHT - (data[i] * (double)HEIGHT / max));
		line[i].color = sf::Color::White;
	}

	window->draw(line);
}

int main(int argc, char **argv)
{
	std::ifstream fin;
	fin.open("latency_log(Stessed).csv");
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
