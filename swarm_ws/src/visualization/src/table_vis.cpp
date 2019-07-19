#include <ros/ros.h>

#include <wvu_swarm_std_msgs/vicon_point.h>
#include <wvu_swarm_std_msgs/vicon_points.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <wvu_swarm_std_msgs/obstacle.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>

#include <std_msgs/String.h>

#include <visualization/contour.h>
#include <visualization/perspective_transform_gpu.h>
#include <visualization/visualization_settings.h>
#include <visualization/keyboard_managment.h>

#include <contour_node/level_description.h>

#include <SFML/Config.hpp>

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
static int g_draw_level;

// turns on verbose mode
#define TAB_DEBUG 0

ContourMap *cont; // contour plot pointer

// sprite that is drawn to screen
sf::Sprite displaySprite;

// other graphical objects
sf::Text rid_disp;
sf::Font comic_sans;

// quad to transform into
quadrilateral_t g_trap;

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

// current contour map data
wvu_swarm_std_msgs::map_levels map;

// Human interaction points
std::vector<sf::Vector2f> nui_points;

sf::Vector2f convertCoordinate(sf::Vector2f a)
{
	sf::Vector2f tab_vect(a.y + table_origin.x, a.x + table_origin.y);

	//tab_vect = tab_vect + table_origin;
	tab_vect.x *= (double) WIDTH / g_table_width;
	tab_vect.y *= (double) HEIGHT / g_table_height;
	return tab_vect;
}

void nuiUpdate(geometry_msgs::Point msg)
{
	ROS_INFO("Got point: (%lf, %lf)", msg.x, msg.y);
	nui_points.clear();
	nui_points.push_back(convertCoordinate(sf::Vector2f(msg.x, msg.y)));
}

void updateMap(wvu_swarm_std_msgs::map_levels _map)
{
	map = _map;
	interaction::universe = &map;
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
	char *vectr = (char*) malloc(sizeof(char) * strlen(vect.c_str()));
	strcpy(vectr, vect.c_str());
	char *x = (char*) malloc(sizeof(char) * strlen(vect.c_str()));
	char *y = (char*) malloc(sizeof(char) * strlen(vect.c_str()));

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
		if (map.levels.size() > g_draw_level)
		{
#if TAB_DEBUG
		std::cout << "Starting contour calc" << std::endl;
		std::cout << "\nDrawing level:\n" << map.levels[g_draw_level] << std::endl;
#endif
			cont->resemble(map.levels[g_draw_level]);
#if TAB_DEBUG
		std::cout << "Resemble contour calc" << std::endl;
#endif
			cont->tick(); // telling the contour plot to advance
#if TAB_DEBUG
		std::cout << "Finished contour calc" << std::endl;
#endif
		}
	}
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
//	std::cout << "Starting render" << std::endl;
#endif
	// creating render texture
	sf::RenderTexture disp;
	disp.create(WIDTH, HEIGHT);

	if (strcmp(g_background.c_str(), "Contour") == 0)
	{
		if (map.levels.size() > g_draw_level)
			cont->render(&disp); // drawing contour plot
	}
	else if (strcmp(g_background.c_str(), "None") == 0)
	{
	}
	else
	{
		// drawing image from file
		sf::Image checker;
		checker.loadFromFile(g_background);
		sf::Texture checker_texture;
		checker_texture.loadFromImage(checker);
		sf::Sprite checker_sprite;
		checker_sprite.setTexture(checker_texture);
		checker_sprite.setPosition(sf::Vector2f(0, 0));
		checker_sprite.scale((double) WIDTH / checker.getSize().x,
				(double) HEIGHT / checker.getSize().y);
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

	// drawing NUI points
	sf::CircleShape nui_circ;
	nui_circ.setRadius(5);
	nui_circ.scale(1, 1.25);
	nui_circ.setFillColor(sf::Color::Cyan);
	nui_circ.setOutlineColor(sf::Color::Black);
	nui_circ.setOutlineThickness(1);
	for (size_t i = 0; i < nui_points.size(); i++)
	{
		nui_circ.setPosition(nui_points.at(i) - sf::Vector2f(1, 1));
		disp.draw(nui_circ);
	}

	// drawing robots
	sf::CircleShape bot;
	float radius = g_robot_diameter * 4.0f; // 4 is effectively '/ 2.0 * 1080 / 100
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
	sf::Uint8 *tf_cols = (sf::Uint8*) malloc(4 * WIDTH * HEIGHT);

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
	bool use_keyboard;

	// getting parameters from launch file to get asset and config locations
	n_priv.param < std::string
			> ("asset_path", assets, "/home/ssvnormandy/git/reu-swarm-ros/swarm_ws/src/visualization/assets");
	n_priv.param < std::string
			> ("config_path", config, "/home/ssvnormandy/git/reu-swarm-ros/swarm_ws/src/visualization/cfg/calib.config");
	n_priv.param < std::string
			> ("background", g_background, assets + "/HockeyRink.png");
	n_priv.param<int>("table_width", g_table_width, 200);
	n_priv.param<int>("table_height", g_table_height, 100);
	n_priv.param<int>("robot_diameter", g_robot_diameter, 5);
	n_priv.param<int>("draw_level", g_draw_level, map_ns::COMBINED);
	n_priv.param<bool>("use_keyboard", use_keyboard, false);

	// setting origin
	table_origin = sf::Vector2f(g_table_width / 2, g_table_height / 2);

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

	// subscribing to draw the NUI intersect onto the table
	ros::Subscriber nui_tracking = n.subscribe("hand_1", 1000, nuiUpdate);

	ros::Subscriber map = n.subscribe("/map_data", 1000, updateMap);

	if (use_keyboard)
	{
		interaction::add_pub = n.advertise < wvu_swarm_std_msgs::obstacle
				> ("/add_obstacle", 1000);
		interaction::rem_pub = n.advertise < std_msgs::String
				> ("/rem_obstacle", 1000);
		interaction::loc_pub = n.advertise < geometry_msgs::Point
				> ("hand_1", 1000);
	}

	// calibrating from file
	calibrateFromFile(config);
	interaction::table = g_trap;

	// default color map setup
	std::vector<double> c_levs;
	c_levs.push_back(0);
	c_levs.push_back(4);
	c_levs.push_back(8);
	c_levs.push_back(12);
	c_levs.push_back(16);
	c_levs.push_back(20);

	std::vector<int> c_reds;
	c_reds.push_back(255);
	c_reds.push_back(255);
	c_reds.push_back(0);
	c_reds.push_back(0);
	c_reds.push_back(0);
	c_reds.push_back(255);

	std::vector<int> c_greens;
	c_greens.push_back(0);
	c_greens.push_back(255);
	c_greens.push_back(255);
	c_greens.push_back(255);
	c_greens.push_back(0);
	c_greens.push_back(0);

	std::vector<int> c_blues;
	c_blues.push_back(0);
	c_blues.push_back(0);
	c_blues.push_back(0);
	c_blues.push_back(255);
	c_blues.push_back(255);
	c_blues.push_back(255);

	// getting from parameter list
	std::vector<double> pc_levs;
	n_priv.getParam("color_levels", pc_levs);

	std::vector<int> pc_reds;
	n_priv.getParam("color_reds", pc_reds);

	std::vector<int> pc_greens;
	n_priv.getParam("color_greens", pc_greens);

	std::vector<int> pc_blues;
	n_priv.getParam("color_blues", pc_blues);

	// checking parameters
	size_t num_colors = pc_levs.size();
	if (pc_reds.size() != num_colors || pc_greens.size() != num_colors
			|| pc_blues.size() != num_colors)
	{
		ROS_ERROR("Number of colors mismatch in launch file");
		throw "MISMATCH";
	}

	// setting parameters
	if (num_colors >= 2)
	{
		c_levs = pc_levs;
		c_reds = pc_reds;
		c_greens = pc_greens;
		c_blues = pc_blues;
	}

	num_colors = c_levs.size();

	// setting up color map from parameters
	ColorMap cmap(
			std::pair<double, sf::Color>(c_levs[0],
					sf::Color(c_reds[0], c_greens[0], c_blues[0])),
			std::pair<double, sf::Color>(c_levs[1],
					sf::Color(c_reds[1], c_greens[1], c_blues[1])));
	for (size_t i = 2; i < num_colors; i++)
	{
		cmap.addColor(
				std::tuple<double, sf::Color>(c_levs[i],
						sf::Color(c_reds[i], c_greens[i], c_blues[i])));
#if TAB_DEBUG
		std::cout << "Adding color: [Lev: " << c_levs[i] << ", Col: " << c_reds[i] << "," << c_greens[i] << "," << c_blues[i] << "]" << std::endl;
#endif
	}
	cont = new ContourMap(sf::Rect<int>(0, 0, WIDTH, HEIGHT), cmap);

	// adding levels
	int num_levels;
	double range_min, range_max;
	n_priv.param<int>("num_levels", num_levels, 9);
	n_priv.param<double>("range_top", range_max, 20);
	n_priv.param<double>("range_bottom", range_min, -20);
	double incr = (range_max - range_min) / (double) num_levels;
	for (double i = range_min; i <= range_max; i += incr)
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

			case sf::Event::KeyReleased:
				if (use_keyboard)
					interaction::keyEvent(event);
				break;

			case sf::Event::MouseButtonPressed:
				if (use_keyboard)
					interaction::mousePressedEvent(event);
				break;

			case sf::Event::MouseButtonReleased:
				if (use_keyboard)
					interaction::mouseReleasedEvent(event);
				break;

			case sf::Event::MouseMoved:
				if (use_keyboard)
					interaction::mouseMovedEvent(event);
				break;

			case sf::Event::MouseWheelScrolled:
				if (use_keyboard)
					interaction::scrollWheelMoved(event);
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
