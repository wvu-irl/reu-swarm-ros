/*
 *  Compile using
 *
 *      g++ main.cpp -oContour.o -lsfml-graphics -lsfml-window -lsfml-system
 */

#include <ros/ros.h>

#include <visualization/contour.h>
#include <visualization/color_map.h>
#include <math.h>
#include <SFML/Graphics.hpp>

#define WIDTH 1200
#define HEIGHT 600

ContourMap *cont;

static int g_tick = 0;

double function(double x, double y)
{
	x -= 600;
	y -= 300;
	x /= 50;
	y /= 50;
	return (x * x - y * y) * sin(M_PI / 50 * g_tick);
}

void tick()
{
	cont->tick();

	g_tick++;
	g_tick %= 100;
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

	//cmap.addColor(std::tuple<double, sf::Color>(0, sf::Color::White));

	cmap.addColor(std::tuple<double, sf::Color>(-6.66667, sf::Color::Yellow));
	cmap.addColor(std::tuple<double, sf::Color>(-3.33333, sf::Color::Green));
	cmap.addColor(std::tuple<double, sf::Color>(3.33333, sf::Color::Cyan));
	cmap.addColor(std::tuple<double, sf::Color>(6.66667, sf::Color::Blue));

	cont = new ContourMap(sf::Rect<int>(0, 0, 1200, 600), cmap);

	int j = 0;
	const double num_levels = 9.0;
	for (double i = -10.0; i <= 10.0; i += 20.0 / num_levels)
	{
		cont->levels.push_back(i);
		std::cout << "Added level: " << i << std::endl;
	}
	cont->resemble(function);

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
	}
}
