#ifndef CONTOUR_H
#define CONTOUR_H

#include <SFML/Graphics.hpp>
#include <vector>
#include "color_map.h"
#include <functional>

#define LINE_CHECK_DIST 1

/**
 * Class representing a 3D Vector and some operations that can be done on it
 */
class Vector3D
{
public:
	// data
	double x, y, z;

	// constructor
	// Starts with x y z because they have to exist for the vector to exist
	Vector3D(double x, double y, double z);

	// gets the magnitude of the vector
	double magnitude() const;

	/**
	 * Performs the dot operation between two vectors
	 */
	double dot(const Vector3D &rhs) const;

	// subtracting vectors
	Vector3D operator-(const Vector3D &rhs) const
	{
		return Vector3D(x - rhs.x, y - rhs.y, z - rhs.z);
	}

	// multiplying a vector by a constant
	Vector3D operator*(double rhs) const
	{
		return Vector3D(rhs * x, rhs * y, rhs * z);
	}

	// aknowledgement of an operator that prints out vectors
	friend std::ostream &operator<<(std::ostream &, const Vector3D &);
};

/**
 * Data visualizer that is able to display contour lines.
 *
 * This visualizer is able to display a specified number of contour lines on a 3D surface, represented by an arbitrary function.
 *
 * The visualizer is able to be realized through SFML drawing to a window.
 *
 */
class ContourMap
{
private:
	// image dataset
	sf::Uint8 *cols;
	// current frame
	sf::Image img;
	// current sprite texture (also the image)
	sf::Texture tex;
	// current sprite (has the texture)
	sf::Sprite sprite;

	// the function that the map is following
	std::function<double(double, double)> zfunc;

	// the color map that the function is using
	// can be applied to the background or the lines themselves
	ColorMap color_mapping;

public:
	// location of the visualizer
	sf::Rect<int> bounds;
	// Levels the contours need to draw
	std::vector<double> levels;

	/**
	 * Constructor
	 *
	 * requires location and colors to draw
	 */
	ContourMap(sf::Rect<int> bounds, ColorMap);

	// calculation and darwing functions
	void tick(int);
	void render(sf::RenderWindow *window);

	// sets the function the plot is to resemble
	void resemble(std::function<double(double, double)> z);

	// scales the image of the plot by a fractional ammount
	void scale(float sx, float sy);

	// accesors for the color map
	void setColorMap(ColorMap);
	ColorMap *getColorMap();

	// destructor
	~ContourMap();
};
//#include "contour.cu"
#endif
