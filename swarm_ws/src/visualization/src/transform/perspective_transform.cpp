#include "../Eigen/Dense"
#include <SFML/Graphics.hpp>
#include <iostream>

typedef struct
{
	sf::Vector2f tl; // 0
	sf::Vector2f tr; // 1
	sf::Vector2f br; // 2
	sf::Vector2f bl; // 3
} quadrilateral_t;

//typedef struct
//{
//	Eigen::MatrixXd coeffs;
//	double w;
//} perspective_t;
//
//Eigen::MatrixXd calcCoeffs(quadrilateral_t &screen)
//{
//	float x[] = { screen.tl.x, screen.tr.x, screen.br.x, screen.bl.x };
//	float y[] = { screen.tl.y, screen.tr.y, screen.br.y, screen.bl.y };
//
//	sf::Vector2f d1(x[1] - x[2], y[1] - y[2]);
//	sf::Vector2f d2(x[3] - x[2], y[3] - y[2]);
//	sf::Vector2f d3(x[0] - x[1] + x[2] - x[3], y[0] - y[1] + y[2] - y[3]);
//
//	Eigen::MatrixXd coeffMat(3, 3);
//	double denom = d1.x * d2.y - d1.y * d2.x;
//	coeffMat(0, 2) = (d3.x * d2.y - d3.y * d2.x) / denom;
//	coeffMat(1, 2) = (d1.x * d3.y - d1.y * d3.x) / denom;
//
//	coeffMat(0, 0) = x[1] - x[0] + coeffMat(0, 2) * x[1];
//	coeffMat(1, 0) = x[3] - x[0] + coeffMat(1, 2) * x[3];
//	coeffMat(2, 0) = x[0];
//	coeffMat(0, 1) = y[1] - y[0] + coeffMat(0, 2) * y[1];
//	coeffMat(1, 1) = y[3] - y[0] + coeffMat(1, 2) * y[3];
//	coeffMat(2, 1) = y[0];
//	coeffMat(2, 2) = 0.0;
//	coeffMat(1, 2) = y[0];
//
//	return coeffMat;
//}
//
//perspective_t getTransform(quadrilateral_t &screen)
//{
//	Eigen::MatrixXd coeff = calcCoeffs(screen);
//	Eigen::MatrixXd trans(3,3);
//	for (int i = 0;i < 3;i++)
//	{
//		for (int j = 0;j < 3;j++)
//		{
//			Eigen::MatrixXd sub(2,2);
//			for (int k = 0;k < 3;k++)
//			{
//				if (k == i)
//					k++;
//				for (int l = 0;l < 3;l++)
//				{
//					if (l == j)
//						l++;
//					sub(k - k > i ? 1 : 0, l - l > j ? 1 : 0) = coeff(i, j);
//				}
//			}
//			trans(i, j) = sub.determinant();
//		}
//	}
//	double w = (trans * Eigen::Vector3d(1,1,1))(2);
//	return (perspective_t) { trans, w } ;
//}

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
//	Eigen::Vector3d uv1 = pers.coeffs * Eigen::Vector3d(initial.x, initial.y, pers.w);
//	return sf::Vector2f(uv1(0), uv1(1));
	sf::Vector2f l_centre = trap.tl && trap.tr && trap.br && trap.bl;
//	std::cout << l_centre.x << " " << l_centre.y << std::endl;
	sf::Vector2f g_centre = sf::Vector2f(width / 2, height / 2);

	quadrilateral_t g_pol = { toPolar(sf::Vector2f(-g_centre.x, -g_centre.y)),
			toPolar(sf::Vector2f(width / 2, -g_centre.y)), toPolar(
					sf::Vector2f(width / 2, height / 2)), toPolar(
					sf::Vector2f(-g_centre.x, height / 2)) };
	quadrilateral_t l_pol = { toPolar(trap.tl - l_centre), toPolar(
			trap.tr - l_centre), toPolar(trap.br - l_centre), toPolar(
			trap.bl - l_centre) };
	sf::Vector2f g_left, g_right, l_left, l_right;

	sf::Vector2f g_init_pol = toPolar(initial - g_centre);

	sf::Vector2f g_sectors[] = { g_pol.tl, toPolar(sf::Vector2f(width / 2, 0)),
			g_pol.tr, toPolar(sf::Vector2f(width, height / 2)), g_pol.br, toPolar(
					sf::Vector2f(width / 2, height)), g_pol.bl, toPolar(
					sf::Vector2f(0, height / 2)) };

	sf::Vector2f l_sectors[] = { l_pol.tl, toPolar(trap.tl && trap.tr), l_pol.tr,
			toPolar(trap.tr && trap.br), g_pol.br, toPolar(trap.br && trap.bl),
			g_pol.bl, toPolar(trap.bl && trap.tl) };

	if (isBetween(g_init_pol.y, g_sectors[0].y, g_sectors[7].y))
	{
		g_left = g_sectors[7];
		g_right = g_sectors[0];
		l_left = l_sectors[7];
		l_right = l_sectors[0];
	}
	else
	{
		for (size_t i = 1; i < 8; i++)
		{
			if (isBetween(g_init_pol.y, g_sectors[i - 1].y, g_sectors[i].y))
			{
				g_left = g_sectors[i - 1];
				g_right = g_sectors[i];
				l_left = l_sectors[i - 1];
				l_right = l_sectors[i];
				break;
			}
		}
	}

	sf::Vector2f fin_pol(
			scale(g_init_pol.x, g_left.x, g_right.x, l_left.x, l_right.x),
			scale(g_init_pol.y, g_left.y, g_right.y, l_left.y, l_right.y));


	sf::Vector2f trap_loc = fromPolar(fin_pol);
//	std::cout << trap_loc.x << " " << trap_loc.y << std::endl;
//	std::cout << fin_pol.x << " " << fin_pol.y << std::endl;
//	std::cout << trap_loc.x + l_centre.x << " " << trap_loc.y + l_centre.y << std::endl;
	return trap_loc + l_centre;
}
