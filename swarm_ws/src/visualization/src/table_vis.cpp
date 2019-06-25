#include <ros/ros.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Vector3.h>

#include "../cfg/table_settings.h" // were the settings for what to draw are
#include "contour.h"
#include "transform/perspective_transform_gpu.h"

#include <math.h>
#include <unistd.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#if BACKGROUND == CONTOUR_PLOT
ContourMap *cont; // contour plot pointer
#endif
// sprite that is drawn to screen
sf::Sprite displaySprite;

// quad to transform into
quadrilateral_t g_trap;

// current time value
static int g_tick = 0;

// summation operator for 2 vectors
sf::Vector2f operator+=(sf::Vector2f &a, sf::Vector2f b)
{
	return a = sf::Vector2f(a.x + b.x, a.y + b.y);
}

// globals for drawing bot's locations to the table
std::vector<sf::Vector2f> bots_pos;
const sf::Vector2f table_origin = sf::Vector2f(100, 50); // origin on the table relative to my origin
																												 // in cm?
// subscription callback to update bot locations
void drawBots(wvu_swarm_std_msgs::vicon_bot_array bots)
{
	for (size_t i = 0;i < bots.poseVect.size();i++)
	{
		// getting pose
		geometry_msgs::Vector3 tf = bots.poseVect.at(i).botPose.transform.translation;
		sf::Vector2f plain_vect((float)tf.y, -(float)tf.x);

		// transforming to screen frame
		plain_vect += table_origin;
		plain_vect.x *= WIDTH / 200.0;
		plain_vect.y *= HEIGHT / 100.0;

		// adding to list
		bots_pos.push_back(plain_vect);
	}
}

/**
 * Reads vectors from the calibration CSV
 *
 * vect is a string that fits the format
 *
 * 			<<X>>,<<Y>>
 *
 * where x and y are string representations of floats
 *
 * returns an actual vector as specified in the file
 */
vector2f_t readVector(std::string vect)
{
	// allocating memory for getting individual strings
	char *vectr = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
	strcpy(vectr, vect.c_str());
	char *x = (char *) malloc(sizeof(char) * strlen(vect.c_str()));
	char *y = (char *) malloc(sizeof(char) * strlen(vect.c_str()));

	// separating by commas
	strcpy(x, strtok(vectr, ","));
	strcpy(y, strtok(NULL, ","));\
	// converting to floats
	vector2f_t vector = {(float) strtod(x, NULL), (float) strtod(y, NULL)};

	// freeing data
	free(y);
	free(x);
	free(vectr);
	return vector;
}

/**
 * Sets current calibration to values found in the file from 'path'
 *
 * path is the location of the *.config file
 *
 * file must have 4 comma separated vectors in it to calibrate
 *
 */
void calibrateFromFile(std::string path)
{
	std::ifstream fin;
	fin.open(path);
	if (fin)
	{
		// getting raw strings
		std::string tl;
		std::string tr;
		std::string br;
		std::string bl;
		fin >> tl;
		fin >> tr;
		fin >> br;
		fin >> bl;
#if TAB_DEBUG
		std::cout << tl << "\n" << tr << "\n" << br << "\n" << bl << std::endl;
#endif
		// converting to real vectors
		// and putting them into a calibration struct
		g_trap.tl = readVector(tl);
		g_trap.tr = readVector(tr);
		g_trap.br = readVector(br);
		g_trap.bl = readVector(bl);
	}
}

/**
 * Continuously looping function
 *
 * keeps track of time and tells aggregated data to tick
 */
void tick()
{
#if BACKGROUND == CONTOUR_PLOT
	cont->tick(g_tick); // telling the contour plot to advance
#endif
	g_tick++; // stepping time
	g_tick %= 1000; // 'sawing' time
}

/**
 * Renders all information to the window
 *
 * here a render texture is distributed to aggregates to draw on
 * the render texture is then transformed to fit projector quadrilateral
 */
void render(sf::RenderWindow *window)
{
	// creating render texture
	sf::RenderTexture disp;
	disp.create(WIDTH, HEIGHT);

#if BACKGROUND == CHECKERBOARD
	// drawing checkerboard
	sf::Image checker;
	checker.loadFromFile("src/visualization/assets/Checkerboard.jpg");
	sf::Texture checker_texture;
	checker_texture.loadFromImage(checker);
	sf::Sprite checker_sprite;
	checker_sprite.setTexture(checker_texture);
	checker_sprite.setPosition(sf::Vector2f(0,0));
	checker_sprite.scale(1280.0 / checker.getSize().x, 800.0 / checker.getSize().y);
	disp.draw(checker_sprite);
#endif
#if BACKGROUND == CONTOUR_PLOT
	cont->render(&disp); // drawing contour plot
#endif
#if BACKGROUND == HOCKEY
	// drawing hockey rink
	sf::Image rink;
	rink.loadFromFile("src/visualization/assets/HockeyRink.png");
	sf::Texture rink_texture;
	rink_texture.loadFromImage(rink);
	sf::Sprite rink_sprite;
	rink_sprite.setTexture(rink_texture);
	rink_sprite.setPosition(sf::Vector2f(0,0));
	rink_sprite.scale(1280.0 / rink.getSize().x, 800.0 / rink.getSize().y);
	disp.draw(rink_sprite);
#endif

	// drawing robots
	sf::CircleShape bot;
        int radius = 25;
	bot.setRadius(radius);
	bot.setFillColor(sf::Color::Yellow);
	for (size_t i = 0;i < bots_pos.size();i++)
	{
		bot.setPosition(sf::Vector2f(bots_pos[i].x - radius, bots_pos[i].y - radius));
		disp.draw(bot);
	}
	bots_pos.clear();

	// filling image
	disp.display();

	// converting to an image to transform
	sf::Image img = disp.getTexture().copyToImage();
	sf::Uint8 *tf_cols = (sf::Uint8 *) malloc(4 * WIDTH * HEIGHT);

	// transforming to calibration value
	perspectiveTransform(g_trap, &disp, tf_cols);

	sf::Image tf_img;
	tf_img.create(WIDTH, HEIGHT, tf_cols);

	sf::Texture tex;
	tex.loadFromImage(tf_img);
	displaySprite.setTexture(tex);

	// displying transform
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
	free(tf_cols); // freeing unused data
}

// main
int main(int argc, char **argv)
{
	ros::init(argc, argv, "table_vis");
	ros::NodeHandle n;

	// subscribing to have bot locations
	ros::Subscriber robots = n.subscribe("/vicon_array", 1000, drawBots);

	// calibrating from file
	calibrateFromFile("src/visualization/cfg/calib.config");

#if BACKGROUND == CONTOUR_PLOT
	// setting up contour plot

	// color map setup
	ColorMap cmap(std::pair<double, sf::Color>(-10, sf::Color::Red),
			std::pair<double, sf::Color>(10, sf::Color::Magenta));
	// cmap.addColor(std::tuple<double, sf::Color>(0, sf::Color::White));
	cmap.addColor(std::tuple<double, sf::Color>(-6.66667, sf::Color::Yellow));
	cmap.addColor(std::tuple<double, sf::Color>(-3.33333, sf::Color::Green));
	cmap.addColor(std::tuple<double, sf::Color>(3.33333, sf::Color::Cyan));
	cmap.addColor(std::tuple<double, sf::Color>(6.66667, sf::Color::Blue));
	cont = new ContourMap(sf::Rect<int>(0, 0, 1280, 800), cmap);

	// adding levels
	const double num_levels = 9.0; // number of levels to draw
	for (double i = -10.0; i <= 10.0; i += 20.0 / num_levels)
	{
		cont->levels.push_back(i);
		std::cout << "Added level: " << i << std::endl;
	}
#endif
	// creating render window
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Contour Plotting",
			sf::Style::Default);

	while (window.isOpen()) // main loop
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			switch (event.type) // basic event poll
			{
			case sf::Event::Closed: // close window event
				window.close();
				break;
			}
		}

		tick(); // running calculations

		// drawomg
		if (window.isOpen())
		{
			window.clear();
			render(&window);
			window.display();
		}

		// looking for callbacks
		ros::spinOnce();
		usleep(1000); // rate
	}
}
