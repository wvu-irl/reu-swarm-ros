#include <visualization/keyboard_managment.h>
#include <std_msgs/String.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <geometry_msgs/Point.h>

#include <chrono>

using namespace std::chrono;

#define scale(val, o_min, o_max, n_min, n_max) (((val - o_min) / (o_max - o_min)) * (n_max - n_min) + n_min)
#define distance(v0, v1) (sqrt(pow(v0.x - v1.x, 2) + pow(v0.y - v1.y, 2)))

#define DEBUG 0

#if DEBUG
#include <stdio.h>
#include <iostream>
#endif

static wvu_swarm_std_msgs::obstacle *g_selected;
static double x, y;

enum MODE
{
	None, Rad_x, Rad_y, Theta_off, Amplitude
};

static MODE editMode = None;

sf::Vector2f interaction::getMouseCordinate(sf::Vector2f initial,
		quadrilateral_t trap)
{
	double unified_x = initial.x;
	double unified_y = initial.y;

	unified_x *= 200.0 / (double) WIDTH;
	unified_y *= 100.0 / (double) HEIGHT;
	unified_x -= 100;
	unified_y -= 50;

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
	n_obs.level = USING_LEVEL;

	interaction::add_pub.publish(n_obs);
}

void interaction::keyEvent(sf::Event e)
{
	if (g_selected != NULL)
	{
		switch (e.key.code)
		{
		case sf::Keyboard::Delete: // removing selected function
			std_msgs::String str;
			str.data = g_selected->characteristic.name;
			interaction::rem_pub.publish(str);
			ROS_INFO("Removed: %s", str.data.c_str());
			break;
		}
	}
	else
	{
		switch (e.key.code)
		{
		case sf::Keyboard::Key::Return: // adding new function
			addNewFunk();
			break;

		case sf::Keyboard::Key::T:
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
			editMode = None;
			g_selected = NULL;
			break;
		}
	}
}

void interaction::mousePressedEvent(sf::Event e)
{
#if DEBUG
	puts("Mouse pressed");
#endif

	if (g_selected != NULL)
	{
		free(g_selected);
	}

	sf::Vector2f mouse(x, y);

	wvu_swarm_std_msgs::obstacle *closest =
			(wvu_swarm_std_msgs::obstacle*) malloc(
					sizeof(wvu_swarm_std_msgs::obstacle));

	closest->characteristic =
			interaction::universe->levels[map_ns::COMBINED].functions[0];

	closest->level = 0;

	sf::Vector2f ellip(-100000, -100000);

	int level = -1, funk = -1;

	for (size_t i = 0; i < interaction::universe->levels.size(); i++)
	{
		for (size_t j = 0; j < interaction::universe->levels.at(i).functions.size();
				j++)
		{
			sf::Vector2f tst(
					interaction::universe->levels[i].functions[j].ellipse.offset_x,
					interaction::universe->levels[i].functions[j].ellipse.offset_y);
			if (funk == -1 || distance(ellip, mouse) > distance(tst, mouse))
			{
				closest->characteristic = interaction::universe->levels[i].functions[j];
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
			interaction::universe->levels[level].functions[funk];

#if DEBUG
	puts("Set function");
#endif
	g_selected->level = level;
}

void interaction::mouseReleasedEvent(sf::Event e)
{

}

void interaction::mouseMovedEvent(sf::Event e)
{
	sf::Vector2f mloc = interaction::getMouseCordinate(
			sf::Vector2f(e.mouseMove.x, e.mouseMove.y), interaction::table);

	x = mloc.x;
	y = mloc.y;

	geometry_msgs::Point pnt;
	pnt.x = x;
	pnt.y = y;
	pnt.z = 0;

	interaction::loc_pub.publish(pnt);

	if (g_selected != NULL)
	{
		g_selected->characteristic.ellipse.offset_x = x;
		g_selected->characteristic.ellipse.offset_y = y;
		interaction::add_pub.publish(*g_selected);

#if DEBUG
		printf("Moved %s to (%0.3lf, %0.3lf)\n", g_selected->characteristic.name.c_str(), x, y);
#endif
	}
}

void interaction::scrollWheelMoved(sf::Event e)
{

}

void interaction::drawSelection(sf::RenderTexture *window)
{

}
