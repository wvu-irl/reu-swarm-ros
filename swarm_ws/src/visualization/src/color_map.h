#ifndef COLOR_MAP_H
#define COLOR_MAP_H

#include <SFML/Graphics.hpp>
#include <vector>

/**
 * 		The color map class is designed to make a gradient of various colors with respective levels.
 *
 *		The gradient is continuous for the entire domain.
 *		If the value exceeds or subseeds the max and min the color will be that of the max or min respectively
 *
 */
class ColorMap
{
public:
	// contains all the number color sets
	std::vector<std::tuple<double, sf::Color>> colors;

	// the map must start out with a max and min
	ColorMap(std::tuple<double, sf::Color> min,
			std::tuple<double, sf::Color> max);

	// add color adds a color to the list
	// adding them in order is not necessary
	void addColor(std::tuple<double, sf::Color>);

	// returns the graduated color value from the assembled gradiant
	sf::Color calculateColor(double val);

};
//#include "color_map.cu"
#endif
