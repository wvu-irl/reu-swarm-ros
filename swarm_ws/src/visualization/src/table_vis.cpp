#include "contour.h"
#include "transform/perspective_transform_gpu.h"

#include <math.h>
#include <unistd.h>
#include <string.h>

#include <iostream>
#include <fstream>

#define WIDTH 1280
#define HEIGHT 800

#define TAB_DEBUG 0

#define CHECKERBOARD_ON 0

ContourMap *cont;
sf::Sprite displaySprite;

//perspective_t g_perspec;
quadrilateral_t g_trap;

static int g_tick = 0;

vector2f_t readVector(std::string vect)
{
	char *vectr = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
	strcpy(vectr, vect.c_str());
	char *x = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
	char *y = (char *) malloc(sizeof(char) * strlen(vect.c_str()));

	strcpy(x, strtok(vectr, ","));
	strcpy(y, strtok(NULL, ","));
	vector2f_t vector = {(float) strtod(x, NULL), (float) strtod(y, NULL)};

	free(y);
	free(x);
	free(vectr);
	return vector;
}

void calibrateFromFile(std::string path)
{
	std::ifstream fin;
	fin.open(path);
	if (fin)
	{
		std::string tl;
		std::string tr;
		std::string br;
		std::string bl;
		fin >> tl;
		fin >> tr;
		fin >> br;
		fin >> bl;

		std::cout << tl << "\n" << tr << "\n" << br << "\n" << bl << std::endl;

		g_trap.tl = readVector(tl);
		g_trap.tr = readVector(tr);
		g_trap.br = readVector(br);
		g_trap.bl = readVector(bl);
	}
}

void tick()
{
#if !CHECKERBOARD_ON
	cont->tick(g_tick);
#endif
	g_tick++;
	g_tick %= 1000;
}

void render(sf::RenderWindow *window)
{
	sf::RenderTexture disp;
	disp.create(WIDTH, HEIGHT);
#if CHECKERBOARD_ON
	sf::Image checker;
	checker.loadFromFile("src/visualization/assets/Checkerboard.jpg");
	sf::Texture checker_texture;
	checker_texture.loadFromImage(checker);
	sf::Sprite checker_sprite;
	checker_sprite.setTexture(checker_texture);
	checker_sprite.setPosition(sf::Vector2f(0,0));
	checker_sprite.scale(1280.0 / 800.0, 1.0);
	disp.draw(checker_sprite);
#else
	cont->render(&disp);
#endif
	disp.display();

	sf::Image img = disp.getTexture().copyToImage();
	sf::Uint8 *tf_cols = (sf::Uint8 *) malloc(4 * WIDTH * HEIGHT);

	perspectiveTransform(g_trap, &disp, tf_cols);

	sf::Image tf_img;
	tf_img.create(WIDTH, HEIGHT, tf_cols);

	sf::Texture tex;
	tex.loadFromImage(tf_img);
	displaySprite.setTexture(tex);

	displaySprite.setPosition(sf::Vector2f(0, 0));

	window->draw(displaySprite);
#if TAB_DEBUG
	sf::VertexArray trapezoid(sf::LineStrip, 5);
	trapezoid[0].position = g_trap.tl;
	trapezoid[1].position = g_trap.tr;
	trapezoid[2].position = g_trap.br;
	trapezoid[3].position = g_trap.bl;
	trapezoid[4].position = g_trap.tl;
	window->draw(trapezoid);
#endif
	free(tf_cols);
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
	calibrateFromFile("src/visualization/cfg/calib.config");
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
