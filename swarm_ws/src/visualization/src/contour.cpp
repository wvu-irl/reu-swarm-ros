#ifndef CONTOUR_CPP
#define CONTOUR_CPP
#include <iostream>
#include <sstream>
#include <math.h>
#include <thread>
#include <future>
#include "contour.h"

#define DEBUG_CONT_SRC 0

// switches between drawing a heat map in the background and
// making the contours use the color mapping
#define SHOW_HEAT_MAP 0

// constructor
Vector3D::Vector3D(double x, double y, double z) :
		x(x), y(y), z(z)
{
}

// calculates the magnitude
double Vector3D::magnitude() const
{
	return sqrt(x * x + y * y + z * z);
}

// does the dot product of this and another vector
double Vector3D::dot(const Vector3D &rhs) const
{
	return x * rhs.x + y * rhs.y + z * rhs.z;
}

// cout operator
std::ostream &operator<<(std::ostream &os, const Vector3D &f)
{
	std::stringstream ss;
	ss << "(" << f.x << ", " << f.y << ", " << f.z << ")";
	return os << ss.str();
}

// Contour map definiton
// constructor
ContourMap::ContourMap(sf::Rect<int> _bounds, ColorMap cm) :
		color_mapping(cm) // setting the color map
{
	bounds = sf::Rect<int>(_bounds); // setting the bounds
	cols = (sf::Uint8 *) malloc(bounds.width * bounds.height * 4); // allocates memory for the image
	img.create(bounds.width, bounds.height, cols); // initializes
	sprite.setPosition(bounds.left, bounds.top); // sets the sprite position
}

void ContourMap::scale(float sx, float sy)
{
	this->sprite.scale(sf::Vector2f(sx, sy)); // scales the sprite
}

// destructor
ContourMap::~ContourMap()
{
	free(cols); // frees image memory
}

// setter for the function
void ContourMap::resemble(std::function<double(double, double)> z)
{
#if DEBUG_CONT_SRC
    std::cout << "setting funk" << std::endl;
#endif
	this->zfunc = z;
}

/**
 * Helper function that returns true if mid is between l and r
 */
bool isBtw(double l, double r, double mid)
{
	return (l < r ? l : r) <= mid && mid <= (l < r ? r : l);
}

// returns the pixel draw location for the map
Vector3D *drawPix(Vector3D points[5], double level)
{
	Vector3D *draw = NULL;

	for (size_t i = 1; i < 5; i++)
	{
		// draws if the level between the center point and one of the neighbors
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
		// for every pixel
		for (size_t i = 0; i < bounds.height; i++)
		{
			for (double j = 0; j < bounds.width; j++)
			{
				// calculates the middle vector
				Vector3D zc = Vector3D(j, i, zfunc(j, i));
#if SHOW_HEAT_MAP
				// when drawing the background sets the pixel to the map color
				img.setPixel(j, i, color_mapping.calculateColor(zc.z));
#else
				// clears the pixel when not drawing the heat map
				img.setPixel(j, i, sf::Color::Transparent);
#endif
				// finding neighbor heights
				Vector3D zu = Vector3D(j, i - LINE_CHECK_DIST,
						zfunc(j, i - LINE_CHECK_DIST));
				Vector3D zd = Vector3D(j, i - LINE_CHECK_DIST,
						zfunc(j, i - LINE_CHECK_DIST));
				Vector3D zl = Vector3D(j - LINE_CHECK_DIST, i,
						zfunc(j - LINE_CHECK_DIST, i));
				Vector3D zr = Vector3D(j + LINE_CHECK_DIST, i,
						zfunc(j + LINE_CHECK_DIST, i));

				// creating an array for drawPix
				Vector3D vects[] = { zc, zu, zd, zl, zr };
				for (double k = 0; k < levels.size(); k++)
				{
					Vector3D *draw = drawPix(vects, levels.at(k));
					if (draw != NULL)
					{
						// setting pixel to value if it is drawable
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
		// draws the image to the screen
		tex.loadFromImage(img);
		sprite.setTexture(tex, true);
		window->draw(sprite);
	}
}

// accessors
void ContourMap::setColorMap(ColorMap cm)
{
	color_mapping = cm;
}

ColorMap *ContourMap::getColorMap()
{
	return &color_mapping;
}
#endif
