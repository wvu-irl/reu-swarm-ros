/**
 * 
 * g++ calibrate.cpp -o Calibrate.o -lsfml-graphics -lsfml-window -lsfml-system
 * 
 */

#include <SFML/Graphics.hpp>
#include <unistd.h>
#include <math.h>
#include <vector>

#include <iostream>
#include <fstream>

#define WIDTH 1280
#define HEIGHT 800

#define DEBUG 0

double operator*(sf::Vector2f a, sf::Vector2f b)
{
	return a.x * b.x + a.y * b.y;
}

const sf::String instructions[] = { "Select TOP LEFT", "Select TOP RIGHT",
		"Select BOTTOM RIGHT", "Select BOTTOM LEFT",
		"Press space to write or R to reset" };
static int state = 0;

typedef struct
{
	sf::Vector2f top_left;
	sf::Vector2f top_right;
	sf::Vector2f bottom_right;
	sf::Vector2f bottom_left;
} calibrate;

sf::Font fon;
sf::Text instruct;

calibrate g_calib;

double scaleX, scaleY;

void render(sf::RenderWindow *window)
{
#if DEBUG
	std::cout << "Started Rendering" << std::endl;
#endif
	sf::VertexArray trap(sf::LinesStrip, state + 1 + (state == 3 ? 1 : 0));
#if DEBUG
	std::cout << "Created vertex array size : "
			<< (state + 1 + (state == 3 ? 1 : 0)) << std::endl;
#endif
	switch (state)
	{
	default:
	case 3:
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
	window->draw(trap);
	window->draw(instruct);
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
	instruct.setString(str);
	instruct.setPosition(
			sf::Vector2f(WIDTH / 2 - instruct.getLocalBounds().width / 2,
			HEIGHT / 2));
}

int main(int argc, char **argv)
{
#if DEBUG
	std::cout << "Started" << std::endl;
#endif
	state = 0;
	g_calib.top_left = sf::Vector2f(0, 0);
	g_calib.top_right = sf::Vector2f(0, 0);
	g_calib.bottom_right = sf::Vector2f(0, 0);
	g_calib.bottom_left = sf::Vector2f(0, 0);

	fon.loadFromFile(sf::String("src/visualization/assets/ComicSansMS3.ttf"));
	instruct.setFillColor(sf::Color::White);
	instruct.setFont(fon);
	instruct.setCharacterSize(20);
	changeInstruction(instructions[0]);
#if DEBUG
	std::cout << "Loaded instruction" << std::endl;
#endif
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Projector Calibration",
			sf::Style::Default);
#if DEBUG
	std::cout << "Loaded window" << std::endl;
#endif
	while (window.isOpen())
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
			switch (event.type)
			{
			case sf::Event::Closed:
#if DEBUG
				std::cout << "Close Event" << std::endl;
#endif
				window.close();
				break;
			case sf::Event::MouseMoved:
#if DEBUG
				std::cout << "Mouse moved Event" << std::endl;
#endif
				switch (state)
				{
				case 0:
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
			case sf::Event::Resized:
				scaleX = (float) WIDTH / (float) event.size.width;
				scaleY = (float) HEIGHT / (float) event.size.height;
				break;
			case sf::Event::MouseButtonReleased:
#if DEBUG
				std::cout << "Mouse released" << std::endl;
#endif
				state += state < 4 ? 1 : 0;
				changeInstruction(instructions[state]);
				break;
			case sf::Event::KeyPressed:
#if DEBUG
				std::cout << "Key Press event" << std::endl;
#endif
				if (event.key.code == sf::Keyboard::R)
				{
					state = 0;
					changeInstruction(sf::String("Select TOP LEFT"));
				}
				else if (state == 4 && event.key.code == sf::Keyboard::Space)
				{
					std::ofstream file;
					file.open("src/visualization/cfg/calib.config", std::ios::out);
					if (file)
					{
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
						std::cout << "\033[30;42mDid not write\033[0m" << std::endl;
					}
					changeInstruction("Done, press R to re-calibrate");
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
