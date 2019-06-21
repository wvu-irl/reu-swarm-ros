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

typedef struct
{
	Eigen::MatrixXd coeffs;
	double w;
} perspective_t;

Eigen::MatrixXd calcCoeffs(quadrilateral_t &screen)
{
	float x[] = { screen.tl.x, screen.tr.x, screen.br.x, screen.bl.x };
	float y[] = { screen.tl.y, screen.tr.y, screen.br.y, screen.bl.y };

	sf::Vector2f d1(x[1] - x[2], y[1] - y[2]);
	sf::Vector2f d2(x[3] - x[2], y[3] - y[2]);
	sf::Vector2f d3(x[0] - x[1] + x[2] - x[3], y[0] - y[1] + y[2] - y[3]);

	Eigen::MatrixXd coeffMat(3, 3);
	double denom = d1.x * d2.y - d1.y * d2.x;
	coeffMat(0, 2) = (d3.x * d2.y - d3.y * d2.x) / denom;
	coeffMat(1, 2) = (d1.x * d3.y - d1.y * d3.x) / denom;

	coeffMat(0, 0) = x[1] - x[0] + coeffMat(0, 2) * x[1];
	coeffMat(1, 0) = x[3] - x[0] + coeffMat(1, 2) * x[3];
	coeffMat(2, 0) = x[0];
	coeffMat(0, 1) = y[1] - y[0] + coeffMat(0, 2) * y[1];
	coeffMat(1, 1) = y[3] - y[0] + coeffMat(1, 2) * y[3];
	coeffMat(2, 1) = y[0];
	coeffMat(2, 2) = y[0];
	coeffMat(1, 2) = 0.0;

	return coeffMat;
}

perspective_t getTransform(quadrilateral_t &screen)
{
	Eigen::MatrixXd coeff = calcCoeffs(screen);
	Eigen::MatrixXd trans(3,3);
	for (int i = 0;i < 3;i++)
	{
		for (int j = 0;j < 3;j++)
		{
			Eigen::MatrixXd sub(2,2);
			for (int k = 0;k < 3;k++)
			{
				if (k == i)
					k++;
				for (int l = 0;l < 3;l++)
				{
					if (l == j)
						l++;
					sub(k - k > i ? 1 : 0, l - l > j ? 1 : 0) = coeff(i, j);
				}
			}
			trans(i, j) = sub.determinant();
		}
	}
	double w = 1 / (trans * Eigen::Vector3d(0,0,1))(2);
	return (perspective_t) { trans, w } ;
}

sf::Vector2f warpPoint(perspective_t pers, sf::Vector2f initial)
{
	Eigen::Vector3d uv1 = pers.coeffs * Eigen::Vector3d(initial.x, initial.y, pers.w);
	return sf::Vector2f(uv1(0), uv1(1));
}
