/*
 *  Compile using
 *
 *       nvcc -ccbin g++ table_vis.cu -o Contour.o -lsfml-graphics -lsfml-window -lsfml-system -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70

 */
#include <ros/ros.h>

#include "contour.h"

#include <math.h>
#include <unistd.h>

#define WIDTH 1280
#define HEIGHT 800

ContourMap *cont;

static int g_tick = 0;

void tick()
{
	cont->tick(g_tick);

	g_tick++;
	g_tick %= 1000;
}

void render(sf::RenderWindow *window)
{
	cont->render(window);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "table_vis");
	ros::NodeHandle n;

	ColorMap cmap(std::pair<double, sf::Color>(-10, sf::Color::Red),
			std::pair<double, sf::Color>(10, sf::Color::Magenta));

	// cmap.addColor(std::tuple<double, sf::Color>(0, sf::Color::White));
	cmap.addColor(std::tuple<double, sf::Color>(-6.66667, sf::Color::Yellow));
	cmap.addColor(std::tuple<double, sf::Color>(-3.33333, sf::Color::Green));
	cmap.addColor(std::tuple<double, sf::Color>(3.33333, sf::Color::Cyan));
	cmap.addColor(std::tuple<double, sf::Color>(6.66667, sf::Color::Blue));

	cont = new ContourMap(sf::Rect<int>(10, 10, 1260, 780), cmap);
	const double num_levels = 9.0;
	for (double i = -10.0; i <= 10.0; i += 20.0 / num_levels)
	{
		cont->levels.push_back(i);
		std::cout << "Added level: " << i << std::endl;
	}

	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Contour Plotting",
			sf::Style::Default);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			switch (event.type)
			{
			case sf::Event::Closed:
				window.close();
				break;
			}
		}

		tick();

		if (window.isOpen())
		{
			window.clear();
			render(&window);
			window.display();
		}
		usleep(1000);
	}
}
