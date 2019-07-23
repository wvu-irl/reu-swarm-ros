#include <visualization/keyboard_managment.h>
#include <std_msgs/String.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <geometry_msgs/Point.h>

#include <chrono>

using namespace std::chrono;

#define distance(v0, v1) (sqrt(pow(v0.x - v1.x, 2) + pow(v0.y - v1.y, 2)))

#define DEBUG 0

#if DEBUG
#include <stdio.h>
#include <iostream>
#endif

ros::Publisher *add_pub, *rem_pub, *loc_pub;
wvu_swarm_std_msgs::map_levels *universe;
quadrilateral_t table;

static wvu_swarm_std_msgs::obstacle *g_selected;
static double x, y;

enum MODE
{
	None, Rad_x, Rad_y, Theta_off, Amplitude
};

static MODE editMode = None;

void interaction::init(ros::Publisher *_add_pub, ros::Publisher *_rem_pub,
		ros::Publisher *_loc_pub, wvu_swarm_std_msgs::map_levels *_universe,
		quadrilateral_t _table)
{
	add_pub = _add_pub;
	rem_pub = _rem_pub;
	loc_pub = _loc_pub;
	universe = _universe;
	table = _table;
}

void interaction::updateUni(wvu_swarm_std_msgs::map_levels *_universe)
{
	universe = _universe;
}

sf::Vector2f interaction::getMouseCordinate(sf::Vector2f initial,
		quadrilateral_t trap)
{
	double unified_x = initial.x;
	double unified_y = initial.y;

	unified_x *= 200.0 / (double) WIDTH;
	unified_y *= 100.0 / (double) HEIGHT;
	unified_x -= 200.0 / 2.0;
	unified_y -= 100.0 / 2.0;

	return sf::Vector2f((float) unified_y, (float) unified_x);
}

void addNewFunk()
{
	wvu_swarm_std_msgs::ellipse ellip;
	ellip.x_rad = 5;
	ellip.y_rad = 5;
	ellip.offset_x = x;
	ellip.offset_y = y;
	ellip.theta_offset = 0;

	wvu_swarm_std_msgs::gaussian charac;
	charac.amplitude = 10;
	charac.ellipse = ellip;
	// creating an arbitrary name
	charac.name = "funk_"
			+ std::to_string(
					(long) duration_cast < milliseconds
							> (high_resolution_clock::now().time_since_epoch()).count());
	charac.selected = false;

	wvu_swarm_std_msgs::obstacle n_obs;
	n_obs.characteristic = charac;
	n_obs.level = g_draw_level;

	add_pub->publish(n_obs);
}

void remove()
{
	if (g_selected != NULL)
	{
		std_msgs::String str;
		str.data = g_selected->characteristic.name;
#if DEBUG
		ROS_INFO("Removing: %s", str.data.c_str());
#endif
		rem_pub->publish(str);
		ROS_INFO("Removed: %s", str.data.c_str());
	}
}

void interaction::keyEvent(sf::Event e)
{
	if (g_selected != NULL)
	{
		switch (e.key.code)
		{
		case sf::Keyboard::Delete: // removing selected function
			remove();
			break;

		case sf::Keyboard::Key::Space:
		case sf::Keyboard::Key::Escape:
#if DEBUG
			puts("Exiting control");
#endif
			editMode = None;
			if (g_selected != NULL)
				g_selected->characteristic.selected = false;
			g_selected = NULL;
			break;
		}
	}
	switch (e.key.code)
	{

	case sf::Keyboard::Key::Return: // adding new function
		addNewFunk();
		break;

	case sf::Keyboard::Key::R:
		editMode = Theta_off;
		break;

	case sf::Keyboard::Key::Y:
		editMode = Rad_y;
		break;

	case sf::Keyboard::Key::X:
		editMode = Rad_x;
		break;

	case sf::Keyboard::Key::A:
		editMode = Amplitude;
		break;

	case sf::Keyboard::Key::Space:
	case sf::Keyboard::Key::Escape:
#if DEBUG
		puts("Exiting control");
#endif
		editMode = None;
		if (g_selected != NULL)
			g_selected->characteristic.selected = false;
		g_selected = NULL;
		break;
	}
}

void interaction::mousePressedEvent(sf::Event e)
{
#if DEBUG
	puts("Mouse pressed");
#endif

	if (editMode != None && g_selected != NULL)
	{
		editMode = None;
		if (g_selected != NULL)
			g_selected->characteristic.selected = false;
		g_selected = NULL;
	}

	if (g_selected != NULL)
	{
		free(g_selected);
	}

	sf::Vector2f mouse(x, y);

	wvu_swarm_std_msgs::obstacle *closest =
			(wvu_swarm_std_msgs::obstacle*) malloc(
					sizeof(wvu_swarm_std_msgs::obstacle));

	closest->characteristic =
			universe->levels[map_ns::COMBINED].functions[0];

	closest->level = 0;

	sf::Vector2f ellip(-100000, -100000);

	int level = -1, funk = -1;

	for (size_t i = 0; i < universe->levels.size(); i++)
	{
		for (size_t j = 0; j < universe->levels.at(i).functions.size();
				j++)
		{
			sf::Vector2f tst(
					universe->levels[i].functions[j].ellipse.offset_x,
					universe->levels[i].functions[j].ellipse.offset_y);
			if (funk == -1 || distance(ellip, mouse) > distance(tst, mouse))
			{
				closest->characteristic = universe->levels[i].functions[j];
				closest->level = i;
				ellip = sf::Vector2f(closest->characteristic.ellipse.offset_x,
						closest->characteristic.ellipse.offset_y);
				level = i;
				funk = j;
			}
		}
	}
	free(closest);

	g_selected = (wvu_swarm_std_msgs::obstacle*) malloc(
			sizeof(wvu_swarm_std_msgs::obstacle));

#if DEBUG
	puts("Allocated memory for obstacle");
	printf("Finding function: LVL:%d, F:%d\n", level, funk);
#endif

	g_selected->characteristic =
			universe->levels[level].functions[funk];

#if DEBUG
	puts("Set function");
#endif
	g_selected->level = level;
	g_selected->characteristic.selected = true;
}

void interaction::mouseReleasedEvent(sf::Event e)
{
	if (editMode == None)
	{
		if (g_selected != NULL)
			g_selected->characteristic.selected = false;
		g_selected = NULL;
	}
}

void interaction::mouseMovedEvent(sf::Event e)
{
#if DEBUG
	puts("Mouse moved");
#endif
	sf::Vector2f mloc = interaction::getMouseCordinate(
			sf::Vector2f(e.mouseMove.x, e.mouseMove.y), table);

	x = mloc.x;
	y = mloc.y;

	geometry_msgs::Point pnt;
	pnt.x = x;
	pnt.y = y;
	pnt.z = 0;

	loc_pub->publish(pnt);

	if (g_selected != NULL)
	{
		g_selected->characteristic.ellipse.offset_x = x;
		g_selected->characteristic.ellipse.offset_y = y;
		add_pub->publish(*g_selected);

#if DEBUG
		printf("Moved %s to (%0.3lf, %0.3lf)\n",
				g_selected->characteristic.name.c_str(), x, y);
#endif
	}
}

void interaction::scrollWheelMoved(sf::Event e)
{
	if (g_selected != NULL)
	{
		switch (editMode)
		{
		case Amplitude:
			g_selected->characteristic.amplitude += e.mouseWheelScroll.delta;
			break;
		case Rad_x:
			g_selected->characteristic.ellipse.x_rad += e.mouseWheelScroll.delta;
			break;
		case Rad_y:
			g_selected->characteristic.ellipse.y_rad += e.mouseWheelScroll.delta;
			break;
		case Theta_off:
			g_selected->characteristic.ellipse.theta_offset +=
					(double) e.mouseWheelScroll.delta / 50.0;
			break;
		default:
			return;
		}
		add_pub->publish(*g_selected);
	}
}
