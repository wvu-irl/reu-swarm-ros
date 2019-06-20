#include "contour.h"
#include "transform/perspective_transform.cpp"

#include <math.h>
#include <unistd.h>
#include <string.h>

#include <iostream>
#include <fstream>

#define WIDTH 1280
#define HEIGHT 800

ContourMap *cont;
sf::Sprite displaySprite;

static int g_tick = 0;

void calibrateFromFile(std::string path)
{
	std::ifstream fin;
	fin.open(path);
	if (fin)
	{
		std::string vect;
		std::string ang;
		std::string wid;
		std::string hei;
		fin >> vect;
		fin >> ang;
		fin >> wid;
		fin >> hei;

		std::cout << vect << "\n" << ang << "\n" << wid << "\n" << hei << std::endl;

		char *vectr = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
		strcpy(vectr, vect.c_str());
		char *x = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
		char *y = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
		strcpy(x, strtok(vectr, ","));
		strcpy(y, strtok(NULL, ","));

		sf::Vector2f pos(strtod(x, NULL), strtod(y, NULL));
		float angle = (float) strtod(ang.c_str(), NULL);
		float width = (float) strtod(wid.c_str(), NULL);
		float height = (float) strtod(hei.c_str(), NULL);

		displaySprite.rotate(angle);
		displaySprite.setPosition(pos);
		displaySprite.scale(sf::Vector2f(width / WIDTH, height / HEIGHT));
		free(vectr);
		free(x);
		free(y);
		fin.close();
	}
}

void tick()
{
	cont->tick(g_tick);

	g_tick++;
	g_tick %= 1000;
}

void render(sf::RenderWindow *window)
{
	sf::RenderTexture disp;
	disp.create(WIDTH, HEIGHT);
	cont->render(&disp);
	disp.display();
	displaySprite.setTexture(disp.getTexture());
	window->draw(displaySprite);
}

int main(int argc, char **argv)
{
	ColorMap cmap(std::pair<double, sf::Color>(-10, sf::Color::Red),
			std::pair<double, sf::Color>(10, sf::Color::Magenta));

	// cmap.addColor(std::tuple<double, sf::Color>(0, sf::Color::White));
	cmap.addColor(std::tuple<double, sf::Color>(-6.66667, sf::Color::Yellow));
	cmap.addColor(std::tuple<double, sf::Color>(-3.33333, sf::Color::Green));
	cmap.addColor(std::tuple<double, sf::Color>(3.33333, sf::Color::Cyan));
	cmap.addColor(std::tuple<double, sf::Color>(6.66667, sf::Color::Blue));

	cont = new ContourMap(sf::Rect<int>(0, 0, 1280, 800), cmap);
	calibrateFromFile("calib.config");
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
