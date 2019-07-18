#include <visualization/keyboard_managment.h>
#include <std_msgs/String.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <geometry_msgs/Point.h>

#include <chrono>

using namespace std::chrono;

#define scale(val, o_min, o_max, n_min, n_max) (((val - o_min) / (o_max - o_min)) * (n_max - n_min) + n_min)

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
	int height = HEIGHT;
	int width = WIDTH;

	sf::Vector2f top((float) scale(initial.x, 0.0, width, trap.tl.x, trap.tr.x),
			(float) scale(initial.x, 0.0, width, trap.tl.y, trap.tr.y));
	sf::Vector2f bottom(
			(float) scale(initial.x, 0.0, width, trap.bl.x, trap.br.x),
			(float) scale(initial.x, 0.0, width, trap.bl.y, trap.br.y));
	sf::Vector2f left((float) scale(initial.y, 0.0, height, trap.bl.x, trap.tl.x),
			(float) scale(initial.y, 0.0, height, trap.bl.y, trap.tl.y));
	sf::Vector2f right(
			(float) scale(initial.y, 0.0, height, trap.br.x, trap.tr.x),
			(float) scale(initial.y, 0.0, height, trap.br.y, trap.tr.y));

	// linear intersection
	double m0 = (right.y - left.y) / (right.x - left.x);
	double m1 = (bottom.y - top.y) / (bottom.x - top.x);
	double unified_x =
			top.x != bottom.x && m0 != m1 && left.x != right.x ?
					(top.y - right.y + right.x * m0 - top.x * m1) / (m0 - m1) : top.x;
	double unified_y =
			left.y != right.y ? (m0 * (unified_x - right.x) + right.y) : left.y;

	unified_x *= 200.0 / (double)WIDTH;
	unified_y *= 100.0 / (double)HEIGHT;
	unified_x -= 100;
	unified_y -= 50;
	unified_y *= -1;
//	unified_x *= -1;

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
}

void interaction::scrollWheelMoved(sf::Event e)
{

}

void interaction::drawSelection(sf::RenderTexture *window)
{

}
