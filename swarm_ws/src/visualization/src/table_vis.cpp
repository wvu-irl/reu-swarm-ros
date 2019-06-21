#include "contour.h"
#include "transform/perspective_transform.cpp"

#include <math.h>
#include <unistd.h>
#include <string.h>

#include <iostream>
#include <fstream>

#define WIDTH 1280
#define HEIGHT 800

#define TAB_DEBUG 0

ContourMap *cont;
sf::Sprite displaySprite;

//perspective_t g_perspec;
quadrilateral_t g_trap;

static int g_tick = 0;

sf::Vector2f readVector(std::string vect)
{
	char *vectr = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
	strcpy(vectr, vect.c_str());
	char *x = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
	char *y = (char *) malloc(sizeof(char) * strlen(vect.c_str()));

	strcpy(x, strtok(vectr, ","));
	strcpy(y, strtok(NULL, ","));
	sf::Vector2f vector((float) strtod(x, NULL), (float) strtod(y, NULL));

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

		g_trap =
		{ readVector(tl), readVector(tr), readVector(br), readVector(bl)};

//		g_perspec = getTransform(g_trap);

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

	sf::Image img = disp.getTexture().copyToImage();
	sf::Uint8 *tf_cols = (sf::Uint8 *) malloc(4 * WIDTH * HEIGHT);
	memset(tf_cols, 0, 4 * WIDTH * HEIGHT);
	sf::Image tf_img;
	tf_img.create(WIDTH, HEIGHT, tf_cols);
	for (size_t i = 0; i < HEIGHT * WIDTH; i++)
	{
		int row = i / WIDTH;
		int col = i % WIDTH;
		sf::Vector2f square_pos((float) col, (float) row);
		sf::Vector2f trap_point = warpPoint(g_trap, WIDTH, HEIGHT, square_pos);
#if TAB_DEBUG
//		sf::Vector2f top(scale(square_pos.x, 0, WIDTH, g_trap.tl.x, g_trap.tr.x),
//				scale(square_pos.x, 0, WIDTH, g_trap.tl.y, g_trap.tr.y));
//		sf::Vector2f bottom(scale(square_pos.x, 0, WIDTH, g_trap.bl.x, g_trap.br.x),
//				scale(square_pos.x, 0, WIDTH, g_trap.bl.y, g_trap.br.y));
//		sf::Vector2f left(scale(square_pos.y, 0, HEIGHT, g_trap.bl.x, g_trap.tl.x),
//				scale(square_pos.y, 0, HEIGHT, g_trap.bl.y, g_trap.tl.y));
//		sf::Vector2f right(scale(square_pos.y, 0, HEIGHT, g_trap.bl.x, g_trap.tl.x),
//				scale(square_pos.y, 0, HEIGHT, g_trap.bl.y, g_trap.tl.y));
//
//		sf::VertexArray tb(sf::LineStrip, 2);
//		tb[0].position = top;
//		tb[1].position = bottom;
//		window->draw(tb);
//
//		sf::VertexArray lr(sf::LineStrip, 2);
//		lr[0].position = left;
//		lr[1].position = right;
//		window->draw(lr);

		std::cout << "Got vector: (" << (int) trap_point.x << ","
				<< (int) trap_point.y << ") <- (" << col << ", " << row << ")"
				<< std::endl;
#endif
		sf::Color pix = img.getPixel((int) square_pos.x, (int) square_pos.y);
		if ((int) trap_point.x < WIDTH && (int) trap_point.y < HEIGHT
				&& (int) trap_point.x >= 0 && (int) trap_point.y >= 0)
			tf_img.setPixel((int) trap_point.x, (int) trap_point.y, pix);
	}

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
