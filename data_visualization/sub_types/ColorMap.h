#ifndef COLOR_MAP_H
#define COLOR_MAP_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <map>

class ColorMap
{
private:
    std::vector<std::pair<double, sf::Color>> colors;

    bool gradiant_;

public:
    ColorMap(std::pair<double, sf::Color> min, std::pair<double, sf::Color> max);

    void addColor(std::pair<double, sf::Color>);
    sf::Color calculateColor(double val);

    void isGradiant(bool);
    bool isGradiant() const;
};

#include "ColorMap.cpp"
#endif