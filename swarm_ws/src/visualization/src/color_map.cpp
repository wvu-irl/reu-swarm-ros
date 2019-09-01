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

#ifndef COLOR_MAP_CPP
#define COLOR_MAP_CPP
#include <visualization/color_map.h>
#include <sstream>
#include <iostream>

// if set to 1 debug messages will appear
#define DEBUG_COLORS 0

// Constructor for color map
ColorMap::ColorMap(std::tuple<double, sf::Color> min,
		std::tuple<double, sf::Color> max)
{
	colors.push_back(min);
	colors.push_back(max);
}

// operator that allows colors to be printed when using std::cout
std::ostream &operator<<(std::ostream &st, const sf::Color col)
{
	std::stringstream ss;
	ss << "Crgb:(" << (int) col.r << "," << (int) col.g << "," << (int) col.b
			<< ")";
	return st << ss.str();
}

/**
 * Adds a color to the set
 *
 * \param val is a tuple that represents a pair of number and color
 *
 * these values are sorted by the map
 */
void ColorMap::addColor(std::tuple<double, sf::Color> val)
{
	colors.push_back(val); // adding color
	std::sort(colors.begin(), colors.end(),
			[](std::tuple<double, sf::Color> a, std::tuple<double, sf::Color> b) -> bool
			{ return std::get<0>(a) < std::get<0>(b);}); // sorting colors by the accompanying double
#if DEBUG_COLORS
    for (size_t i = 0;i < colors.size();i++)
    {
        std::cout << std::get<0>(colors[i]) << " & " << std::get<1>(colors[i]) << ",";
    }
    std::cout << std::endl;
#endif

}


/**
 * An internal function that scales a value from one range to another
 *
 * \param old_range is a tuple that contains the range val is currently in, [0] : min & [1] : max
 * \param new_range is a tuple that contains the range val is being transfered to with the same min/max value order
 * \param val is the value to be converted
 *
 * \returns the scaled value of val in the new range
 */
double scale(double val, std::tuple<double, double> old_range,
		std::tuple<double, double> new_range)
{
	double o_max = std::get < 1 > (old_range);
	double n_max = std::get < 1 > (new_range);
	double o_min = std::get < 0 > (old_range);
	double n_min = std::get < 0 > (new_range);

	return (val - o_min) / (o_max - o_min) * (n_max - n_min) + n_min;
}

/**
 * Finds the gradient for the given value
 *
 * \param val the double for the value in the gradiant
 */
sf::Color ColorMap::calculateColor(double val)
{
	// low edge case
	if (val <= std::get < 0 > (colors[0]))
		return std::get < 1 > (colors[0]);

	// high edge case
	if (val >= std::get < 0 > (colors[colors.size() - 1]))
		return std::get < 1 > (colors[colors.size() - 1]);

	// mid case
	for (size_t i = 1; i < colors.size(); i++)
	{
		// finding the first color to have a greater value than the input
		if (std::get < 0 > (colors[i]) > val)
		{
			sf::Color col(0, 0, 0);
			// finding scaled RGB value
			col.r = (int) scale(val,
					std::tuple<double, double>(std::get < 0 > (colors[i]),
							std::get < 0 > (colors[i - 1])),
					std::tuple<double, double>(std::get < 1 > (colors[i]).r,
							std::get < 1 > (colors[i - 1]).r));
			col.g = (int) scale(val,
					std::tuple<double, double>(std::get < 0 > (colors[i]),
							std::get < 0 > (colors[i - 1])),
					std::tuple<double, double>(std::get < 1 > (colors[i]).g,
							std::get < 1 > (colors[i - 1]).g));
			col.b = (int) scale(val,
					std::tuple<double, double>(std::get < 0 > (colors[i]),
							std::get < 0 > (colors[i - 1])),
					std::tuple<double, double>(std::get < 1 > (colors[i]).b,
							std::get < 1 > (colors[i - 1]).b));
			return col;
		}
	}
	return sf::Color::Black;
}
#endif
