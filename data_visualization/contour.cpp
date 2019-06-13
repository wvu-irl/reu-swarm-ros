#include <iostream>
#include <sstream>
#include <math.h>
#include <thread>
#include <future>
#include "contour.h"

#define DEBUG_CONT_SRC 0

#define SHOW_HEAT_MAP 0

Vector3D::Vector3D(double x, double y, double z) :
		x(x), y(y), z(z)
{
}

double Vector3D::magnitude() const
{
	return sqrt(x * x + y * y + z * z);
}

double Vector3D::dot(const Vector3D &rhs) const
{
	return x * rhs.x + y * rhs.y + z * rhs.z;
}

std::ostream &operator<<(std::ostream &os, const Vector3D &f)
{
	std::stringstream ss;
	ss << "(" << f.x << ", " << f.y << ", " << f.z << ")";
	return os << ss.str();
}

// Contour map definiton

ContourMap::ContourMap(sf::Rect<int> _bounds, ColorMap cm) :
		color_mapping(cm)
{
	bounds = sf::Rect<int>(_bounds);
	cols = (sf::Uint8 *) malloc(bounds.width * bounds.height * 4);
	img.create(bounds.width, bounds.height, cols);
	sprite.setPosition(bounds.left, bounds.top);
}

void ContourMap::scale(float sx, float sy)
{
	this->sprite.scale(sf::Vector2f(sx, sy));
}

ContourMap::~ContourMap()
{
	free(cols);
}

void ContourMap::resemble(std::function<double(double, double)> z)
{
#if DEBUG_CONT_SRC
    std::cout << "setting funk" << std::endl;
#endif
	this->zfunc = z;
}

bool isBtw(double l, double r, double mid)
{
	return (l < r ? l : r) <= mid && mid <= (l < r ? r : l);
}

Vector3D *drawPix(Vector3D points[5], double level)
{
	Vector3D *draw = NULL;

	for (size_t i = 1; i < 5; i++)
	{
		if (isBtw(points[0].z, points[i].z, level))
		{
			draw = points;
			break;
		}
	}
	return draw;
}

void ContourMap::tick()
{
	if (levels.size() > 0)
	{
		for (size_t i = 0; i < bounds.height; i++)
		{
			for (double j = 0; j < bounds.width; j++)
			{
				Vector3D zc = Vector3D(j, i, zfunc(j, i));
#if SHOW_HEAT_MAP
                img.setPixel(j, i, color_mapping.calculateColor(zc.z));
#else
				img.setPixel(j, i, sf::Color::Transparent);
#endif
				Vector3D zu = Vector3D(j, i - LINE_CHECK_DIST,
						zfunc(j, i - LINE_CHECK_DIST));
				Vector3D zd = Vector3D(j, i - LINE_CHECK_DIST,
						zfunc(j, i - LINE_CHECK_DIST));
				Vector3D zl = Vector3D(j - LINE_CHECK_DIST, i,
						zfunc(j - LINE_CHECK_DIST, i));
				Vector3D zr = Vector3D(j + LINE_CHECK_DIST, i,
						zfunc(j + LINE_CHECK_DIST, i));

				Vector3D vects[] = { zc, zu, zd, zl, zr };
				for (double k = 0; k < levels.size(); k++)
				{
					Vector3D *draw = drawPix(vects, levels.at(k));
					if (draw != NULL)
					{
#if SHOW_HEAT_MAP
                        img.setPixel((int)draw->x, (int)draw->y, sf::Color::Transparent);
#else
						img.setPixel((int) draw->x, (int) draw->y,
								color_mapping.calculateColor(zc.z));
#endif
						break;
					}
				}
			}
		}
	}
}

void ContourMap::render(sf::RenderWindow *window)
{
	if (levels.size() > 0)
	{
		tex.loadFromImage(img);
		sprite.setTexture(tex, true);
		window->draw(sprite);
	}
}

void ContourMap::setColorMap(ColorMap cm)
{
	color_mapping = cm;
}

ColorMap *ContourMap::getColorMap()
{
	return &color_mapping;
}
