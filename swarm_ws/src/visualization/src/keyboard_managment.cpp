#include <visualization/keyboard_managment.h>

#define scale(val, n_min, n_max, o_min, o_max) (((val - o_min) / (o_max - o_min)) * (n_max - n_min) + n_min)

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

	return sf::Vector2f((float) unified_x, (float) unified_y);
}

void interaction::keyEvent(sf::Event e)
{
	switch (e.key.code)
	{
	case sf::Keyboard::Delete:

		break;
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

}

void interaction::scrollWheelMoved(sf::Event e)
{

}
