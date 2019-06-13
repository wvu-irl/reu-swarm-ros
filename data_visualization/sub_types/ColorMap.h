#ifndef COLOR_MAP_H
#define COLOR_MAP_H

#include <SFML/Graphics.hpp>
#include <vector>

class ColorMap
{
private:
	std::vector<std::tuple<double, sf::Color>> colors;

public:
	ColorMap(std::tuple<double, sf::Color> min,
			std::tuple<double, sf::Color> max);

	void addColor(std::tuple<double, sf::Color>);
	sf::Color calculateColor(double val);
};

#include "ColorMap.cpp"
#endif
