/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#ifndef CONTOUR_H
#define CONTOUR_H

#include <SFML/Graphics.hpp>
#include <vector>
#include "color_map.h"

#include <wvu_swarm_std_msgs/map_level.h>

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
	wvu_swarm_std_msgs::map_level zfunc;

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
	void tick();
	void render(sf::RenderTexture *window);

	// sets the function the plot is to resemble
	void resemble(wvu_swarm_std_msgs::map_level);

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
