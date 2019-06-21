#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>

#define DEBUG 0

typedef struct
{
	sf::Vector2f tl; // 0
	sf::Vector2f tr; // 1
	sf::Vector2f br; // 2
	sf::Vector2f bl; // 3
} quadrilateral_t;

double scale(double val, double o_min, double o_max, double n_min, double n_max)
{
	if (o_max == o_min)
		return n_min;
	return ((val - o_min) / (o_max - o_min)) * (n_max - n_min) + n_min;
}

sf::Vector2f operator&&(sf::Vector2f a, sf::Vector2f b)
{
	return sf::Vector2f((a.x + b.x) / 2, (a.y + b.y) / 2);
}

double distance(sf::Vector2f a, sf::Vector2f b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

sf::Vector2f toPolar(sf::Vector2f a)
{
	return sf::Vector2f(distance(sf::Vector2f(0, 0), a),
			a.x == 0 ?
					(a.y < 0 ? 3 * M_PI_2 : M_PI_2) :
					(atan(a.y / a.x) + (a.x < 0 ? M_PI : 0)));
}

sf::Vector2f fromPolar(sf::Vector2f p)
{
	return sf::Vector2f(p.x * cos(p.y), p.x * sin(p.y));
}

sf::Vector2f operator-(sf::Vector2f a, sf::Vector2f b)
{
	return sf::Vector2f(a.x - b.x, a.y - b.y);
}

sf::Vector2f operator+(sf::Vector2f a, sf::Vector2f b)
{
	return sf::Vector2f(a.x + b.x, a.y + b.y);
}

bool isBetween(double val, double bound_a, double bound_b)
{
	return val <= (bound_a > bound_b ? bound_a : bound_b)
			&& val >= (bound_a > bound_b ? bound_b : bound_a);
}

sf::Vector2f warpPoint(quadrilateral_t trap, size_t width, size_t height,
		sf::Vector2f initial)
{
	sf::Vector2f top(scale(initial.x, 0, width, trap.tl.x, trap.tr.x),
			scale(initial.x, 0, width, trap.tl.y, trap.tr.y));
	sf::Vector2f bottom(scale(initial.x, 0, width, trap.bl.x, trap.br.x),
			scale(initial.x, 0, width, trap.bl.y, trap.br.y));
	sf::Vector2f left(scale(initial.y, 0, height, trap.bl.x, trap.tl.x),
			scale(initial.y, 0, height, trap.bl.y, trap.tl.y));
	sf::Vector2f right(scale(initial.y, 0, height, trap.br.x, trap.tr.x),
			scale(initial.y, 0, height, trap.br.y, trap.tr.y));
#if DEBUG
	std::cout << "\033[31m[top]" << top.x << "," << top.y << " \033[32m[bottom]" << bottom.x << "," << bottom.y
			<< " \033[33m[left]" << left.x << "," << left.y << " \033[34m[right]" << right.x << "," << right.y
			<< "\033[0m" << std::endl;
#endif
	double m0 = (right.y - left.y) / (right.x - left.x);
	double m1 = (bottom.y - top.y) / (bottom.x - top.x);
#if DEBUG
	std::cout << "\033[30;43mm0: " << m0 << " \033[30;42mm1: " << m1 << "\033[0m" << std::endl;
#endif
	double unified_x =
			top.x != bottom.x && m0 != m1 && left.x != right.x ?
					(top.y - right.y + right.x * m0 - top.x * m1) / (m0 - m1) : top.x;
	double unified_y =
			left.y != right.y ? (m0 * (unified_x - right.x) + right.y) : left.y;

	return sf::Vector2f((float) unified_x, (float) unified_y);
}
