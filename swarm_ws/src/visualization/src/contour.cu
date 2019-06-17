#ifndef CONTOUR_CPP
#define CONTOUR_CPP
#include <iostream>
#include <sstream>
#include <math.h>
#include <thread>
#include <future>

#include "calculation.cu"

#include "contour.h"

#define DEBUG_CONT_SRC 0

// switches between drawing a heat map in the background and
// making the contours use the color mapping
#define SHOW_HEAT_MAP 1

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
	sprite.setPosition(bounds.left, bounds.top); // sets the sprite position
	cols = (sf::Uint8 *)malloc(sizeof(sf::Uint8) * bounds.width * bounds.height * 4);
	hyperthread::init();
}

void ContourMap::scale(float sx, float sy)
{
	this->sprite.scale(sf::Vector2f(sx, sy)); // scales the sprite
}

// destructor
ContourMap::~ContourMap()
{
	hyperthread::destruct();
	free(cols);
}

// setter for the function
void ContourMap::resemble(std::function<double(double, double)> z)
{
#if DEBUG_CONT_SRC
    std::cout << "setting funk" << std::endl;
#endif
	this->zfunc = z;
}

void ContourMap::tick(int tick)
{
	hyperthread::color *colors = (hyperthread::color *)malloc(sizeof(hyperthread::color) * color_mapping.colors.size());
	double *c_levels = (double *)malloc(sizeof(double) * color_mapping.colors.size());
	for (size_t i = 0;i < color_mapping.colors.size();i++)
	{
		sf::Color color = std::get<1>(color_mapping.colors[i]);
		colors[i] = (hyperthread::color){color.r, color.g, color.b};
		c_levels[i] = std::get<0>(color_mapping.colors[i]);
	}

	hyperthread::calc(cols, 
		levels, 
		bounds.width, bounds.height, 
		colors, c_levels, color_mapping.colors.size(), tick);

	// for (size_t i = 0;i < bounds.width * bounds.height * 4;i++)
	// {
	// 	std::cout << "COL[" << i << "]: R=" << (int)cols[i] << ", G=" << (int)cols[i + 1] << ", B=" << (int)cols[i + 2] << std::endl;
	// }

	img.create(bounds.width, bounds.height, cols);

	free(colors);
	free(c_levels);
}

void ContourMap::render(sf::RenderWindow *window)
{
	// draws the image to the screen
	tex.loadFromImage(img);
	sprite.setTexture(tex, true);
	window->draw(sprite);
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
