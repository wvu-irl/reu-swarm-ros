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

#define WIDTH 1280
#define HEIGHT 800

double operator*(sf::Vector2f a, sf::Vector2f b)
{
	return a.x * b.x + a.y * b.y;
}

enum STAGE
{
	TOP_LEFT, TOP_RIGHT, HEIGHT_VAL, WRITE
};

static int state = TOP_LEFT;

typedef struct
{
	sf::Vector2f top_left;
	double angle;
	double height;
	double width;
} calibrate;

calibrate g_calib;

void render(sf::RenderWindow *window)
{
	if (g_calib.top_left.x >= 0 && g_calib.top_left.y)
	{
		if (state >= TOP_RIGHT)
		{
			if (g_calib.height > 0)
			{
				sf::RectangleShape box;
				box.setSize(sf::Vector2f(g_calib.width, g_calib.height));
				box.setPosition(g_calib.top_left);
				box.setFillColor(sf::Color::Transparent);
				box.setOutlineColor(sf::Color::White);
				box.setOutlineThickness(1);
				box.rotate(g_calib.angle * 180.0 / M_PI);
				window->draw(box);
			}
			else
			{
				sf::VertexArray line(sf::LinesStrip, 2);
				line[0].position = g_calib.top_left;
				line[1].position = sf::Vector2f(
						g_calib.width * cos(g_calib.angle) + g_calib.top_left.x,
						g_calib.width * sin(g_calib.angle) + g_calib.top_left.y);
				for (size_t i = 0; i < 4; i++)
				{
					line[i].color = sf::Color::White;
				}
				window->draw(line);
			}

		}
		else
		{
			sf::CircleShape point;
			point.setRadius(4);
			point.setPosition(
					sf::Vector2f(g_calib.top_left.x - 2, g_calib.top_left.y - 2));
			point.setFillColor(sf::Color::White);
			window->draw(point);
		}

	}
}

double anglebetween(double x0, double y0, double x1, double y1)
{
	if (x0 == x1)
	{
		if (y0 > y1)
		{
			return -M_PI_2;
		}
		else
		{
			return M_PI_2;
		}
	}
	return atan((y1 - y0) / (x1 - x0)) + (x1 < x0 ? M_PI : 0);
}

void changeInstruction(sf::String str)
{
	std::cout << str.toAnsiString() << std::endl;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "calibrate");
	ros::NodeHandle n;
	g_calib.angle = -1;
	g_calib.height = 0;
	g_calib.width = 0;
	g_calib.top_left = sf::Vector2f(-1, -1);

	changeInstruction(sf::String("Select TOP LEFT"));

	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Projector Calibration",
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
			case sf::Event::MouseMoved:
				switch (state)
				{
				case TOP_LEFT:
					g_calib.top_left.x = event.mouseMove.x;
					g_calib.top_left.y = event.mouseMove.y;
					break;

				case TOP_RIGHT:
					g_calib.angle = anglebetween(g_calib.top_left.x, g_calib.top_left.y,
							event.mouseMove.x, event.mouseMove.y);
					g_calib.width = sqrt(
							pow(g_calib.top_left.x - event.mouseMove.x, 2)
									+ pow(g_calib.top_left.y - event.mouseMove.y, 2));
					break;

				case HEIGHT_VAL:
					g_calib.height = sf::Vector2f(cos(g_calib.angle + M_PI_2),
							sin(g_calib.angle + M_PI_2))
							* sf::Vector2f(event.mouseMove.x - g_calib.top_left.x,
									event.mouseMove.y - g_calib.top_left.y);
					break;
				case WRITE:
					break;
				}
				break;
			case sf::Event::MouseButtonReleased:
				state += state < WRITE ? 1 : 0;
				switch (state)
				{
				case TOP_LEFT:
					changeInstruction(sf::String("Select TOP LEFT"));
					break;
				case TOP_RIGHT:
					changeInstruction(sf::String("Select TOP RIGHT"));
					break;
				case HEIGHT_VAL:
					changeInstruction("Select HEIGHT");
					break;
				case WRITE:
					changeInstruction("Press SPACE to write, press R to re-calibrate");
					break;
				}
				break;
			case sf::Event::KeyPressed:
				if (event.key.code == sf::Keyboard::R)
				{
					g_calib.angle = -1;
					g_calib.height = 0;
					g_calib.width = 0;
					g_calib.top_left = sf::Vector2f(-1, -1);
					state = TOP_LEFT;
					changeInstruction(sf::String("Select TOP LEFT"));
				}
				else if (state == WRITE && event.key.code == sf::Keyboard::Space)
				{
					std::ofstream file;
					file.open("calib.config", std::ios::out);
					file << g_calib.top_left.x << "," << g_calib.top_left.y << std::endl;
					file << g_calib.angle * 180.0 / M_PI << std::endl;
					file << g_calib.width << std::endl;
					file << g_calib.height << std::endl;
					file.close();
					changeInstruction("Done, press R to re-calibrate");
				}
				break;
			}
		}

		if (window.isOpen())
		{
			window.clear();
			render(&window);
			window.display();
		}
		usleep(10000);
	}

	return 0;
}
