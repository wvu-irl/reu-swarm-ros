#include <ros/ros.h>
#include <wvu_swarm_std_msgs/vicon_point.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Vector3.h>

#include <visualization/contour.h>
#include "transform/perspective_transform_gpu.h"

#include <math.h>
#include <unistd.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

// params from launch
static std::string g_background;
static int g_table_width, g_table_height;
static int g_robot_diameter;

// width and height of generated image
// also of starting window size in pixels
#define WIDTH 1280
#define HEIGHT 800

// turns on verbose mode
#define TAB_DEBUG 1

ContourMap *cont; // contour plot pointer

// sprite that is drawn to screen
sf::Sprite displaySprite;

// other graphical objects
sf::Text rid_disp;
sf::Font comic_sans;

// quad to transform into
quadrilateral_t g_trap;

// current time value
static int g_tick = 0;

// summation operator for 2 vectors
sf::Vector2f operator+(sf::Vector2f a, sf::Vector2f b)
{
	return sf::Vector2f(a.x + b.x, a.y + b.y);
}

sf::Vector2f operator-(sf::Vector2f a, sf::Vector2f b)
{
	return sf::Vector2f(a.x - b.x, a.y - b.y);
}

// globals for drawing bot's locations to the table
std::vector<sf::Vector2f> bots_pos;
std::vector<std::string> bot_ids;
sf::Vector2f table_origin; // origin on the table relative to my origin
// in centimeters

// globals for drawing obstacles
std::vector<sf::Vector2f> obstacles;

// globals for drawing targets
std::vector<sf::Vector2f> targets;

sf::Vector2f convertCoordinate(sf::Vector2f a)
{
	sf::Vector2f tab_vect(a.y + table_origin.x, a.x + table_origin.y);

	//tab_vect = tab_vect + table_origin;
	tab_vect.x *= (double) WIDTH / g_table_width;
	tab_vect.y *= (double) HEIGHT / g_table_height;
	return tab_vect;
}

// Subscription callback for goals
void drawGoals(wvu_swarm_std_msgs::vicon_points goals)
{
        targets.clear();
	for (size_t i = 0; i < goals.point.size(); i++)
	{
		wvu_swarm_std_msgs::vicon_point pnt = goals.point.at(i);
		targets.push_back(convertCoordinate(sf::Vector2f(pnt.x, pnt.y)));
	}
}

// obstacle subscription callback
void drawObstacles(wvu_swarm_std_msgs::vicon_points points)
{
        obstacles.clear();
	for (size_t i = 0; i < points.point.size(); i++)
	{
		wvu_swarm_std_msgs::vicon_point pnt = points.point.at(i);
		obstacles.push_back(convertCoordinate(sf::Vector2f(pnt.x, pnt.y)));
	}
}

// subscription callback to update bot locations
void drawBots(wvu_swarm_std_msgs::vicon_bot_array bots)
{
        bots_pos.clear();
	for (size_t i = 0; i < bots.poseVect.size(); i++)
	{
		// getting pose
		geometry_msgs::Vector3 tf =
				bots.poseVect.at(i).botPose.transform.translation;
		sf::Vector2f plain_vect = convertCoordinate(sf::Vector2f(tf.x, tf.y));

		// adding to list
		bots_pos.push_back(plain_vect);
		char idc[3];
		idc[0] = bots.poseVect.at(i).botId[0];
		idc[1] = bots.poseVect.at(i).botId[1];
		idc[2] = '\0';
		bot_ids.push_back(std::string(idc));
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
	strcpy(y, strtok(NULL, ","));
	// converting to floats
	vector2f_t vector = { (float) strtod(x, NULL), (float) strtod(y, NULL) };

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
	if (strcmp(g_background.c_str(), "Contour") == 0)
	{
		cont->tick(g_tick); // telling the contour plot to advance
	}
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
#if TAB_DEBUG
	std::cout << "Starting render" << std::endl;
#endif
	// creating render texture
	sf::RenderTexture disp;
	disp.create(WIDTH, HEIGHT);

	if (strcmp(g_background.c_str(), "Contour") == 0)
	{
		cont->render(&disp); // drawing contour plot
	}
	else if (strcmp(g_background.c_str(), "None") == 0)
	{
	}
	else
	{
		sf::Image checker;
		checker.loadFromFile(g_background);
		sf::Texture checker_texture;
		checker_texture.loadFromImage(checker);
		sf::Sprite checker_sprite;
		checker_sprite.setTexture(checker_texture);
		checker_sprite.setPosition(sf::Vector2f(0, 0));
		checker_sprite.scale((double)WIDTH / checker.getSize().x,
		(double)HEIGHT / checker.getSize().y);
		disp.draw(checker_sprite);
	}

	// draw obstacles
	sf::CircleShape obs;
	obs.setRadius(4);
	obs.scale(1, 1.25);
	obs.setFillColor(sf::Color::Red);
	for (size_t i = 0; i < obstacles.size(); i++)
	{
		obs.setPosition(obstacles.at(i) - sf::Vector2f(2, 2));
		disp.draw(obs);
	}

	// draw targets
	sf::CircleShape tar;
	tar.setRadius(4);
	tar.scale(1, 1.25);
	tar.setFillColor(sf::Color::Green);
	for (size_t i = 0; i < targets.size(); i++)
	{
		tar.setPosition(targets.at(i) - sf::Vector2f(2, 2));
		disp.draw(tar);
	}
	

	// drawing robots
	sf::CircleShape bot;
	float radius = g_robot_diameter * 4.0f; // 4 is effectively '/ 2.0 * 800 / 100
	bot.setRadius(radius);
	bot.scale(1, 1.25); // scaling to  do a 2:1 screen
	bot.setFillColor(sf::Color::Yellow);
	for (size_t i = 0; i < bots_pos.size(); i++)
	{
		bot.setPosition(
				sf::Vector2f(bots_pos[i].x - radius, bots_pos[i].y - radius));
		disp.draw(bot);

		rid_disp.setString(bot_ids.at(i).c_str());
		rid_disp.setPosition(
				sf::Vector2f(bots_pos[i].x + radius, bots_pos[i].y - radius));
		disp.draw(rid_disp);
	}
	

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
	free(tf_cols); // freeing unused data
}

// main
int main(int argc, char **argv)
{
	ros::init(argc, argv, "table_vis");
	ros::NodeHandle n;

	ros::NodeHandle n_priv("~");

        ros::Rate rate(50);
        
	std::string assets, config;

	n_priv.param < std::string
			> ("asset_path", assets, "/home/ssvnormandy/git/reu-swarm-ros/swarm_ws/src/visualization/assets");
	n_priv.param < std::string
			> ("config_path", config, "/home/ssvnormandy/git/reu-swarm-ros/swarm_ws/src/visualization/cfg/calib.config");
	n_priv.param < std::string
			> ("background", g_background, assets + "/HockeyRink.png");
	n_priv.param<int>("table_width", g_table_width, 200);
	n_priv.param<int>("table_height", g_table_height, 100);
	n_priv.param<int>("robot_diameter", g_robot_diameter, 5);
        
        table_origin = sf::Vector2f(g_table_width / 2,
		g_table_height / 2);

#if TAB_DEBUG
	std::cout << "Got params:\n\t" << g_background << "\n\t" << config << "\n\t" << g_table_width
			<< "\n\t" << g_table_height << "\n\t" << g_robot_diameter << std::endl;

#endif

	// Setting up universal graphics objects
	comic_sans.loadFromFile(assets + "/ComicSansMS3.ttf");
	rid_disp.setFont(comic_sans);
	rid_disp.setColor(sf::Color::Black);
	rid_disp.setCharacterSize(18);

	// subscribing to have bot locations
	ros::Subscriber robots = n.subscribe("/vicon_array", 1000, drawBots);
	// subscribing for other objects
	ros::Subscriber obstacles = n.subscribe("virtual_obstacles", 1000,
			drawObstacles);
	ros::Subscriber goals = n.subscribe("virtual_targets", 1000, drawGoals);

	// calibrating from file
	calibrateFromFile(config);

	// setting up contour plotdrawObstacles

	// color map setup
	ColorMap cmap(std::pair<double, sf::Color>(0, sf::Color::Red),
			std::pair<double, sf::Color>(20, sf::Color::Magenta));
	// cmap.addColor(std::tuple<double, sf::Color>(0, sf::Color::White));
	cmap.addColor(std::tuple<double, sf::Color>(4, sf::Color::Yellow));
	cmap.addColor(std::tuple<double, sf::Color>(8, sf::Color::Green));
	cmap.addColor(std::tuple<double, sf::Color>(12, sf::Color::Cyan));
	cmap.addColor(std::tuple<double, sf::Color>(16, sf::Color::Blue));
	cont = new ContourMap(sf::Rect<int>(0, 0, 1280, 800), cmap);

	// adding levels
	const double num_levels = 10.0; // number of levels to draw
	const double range = 25.0;
	for (double i = range / num_levels; i <= range; i += range / num_levels)
	{
		cont->levels.push_back(i);
	}

	// creating render window
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Table Plotting",
			sf::Style::Default);

	while (window.isOpen() && ros::ok()) // main loop
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
		if (window.isOpen() && ros::ok())
		{
			tick(); // running calculations
			window.clear();
			render(&window);
			window.display();
		}

		// looking for callbacks
		ros::spinOnce();
		rate.sleep(); // rate
	}
}
