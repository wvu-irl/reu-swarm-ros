#include "ColorMap.h"
#include <sstream>
#include <iostream>

#define DEBUG_COLORS 0

ColorMap::ColorMap(std::tuple<double, sf::Color> min, std::tuple<double, sf::Color> max)
{
    colors.push_back(min);
    colors.push_back(max);
}

std::ostream &operator<<(std::ostream &st, const sf::Color col)
{
    std::stringstream ss;
    ss << "Crgb:(" << (int)col.r << "," << (int)col.g << "," << (int)col.b << ")";
    return st << ss.str();
}

void ColorMap::addColor(std::tuple<double, sf::Color> val)
{
    colors.push_back(val);
    std::sort(colors.begin(), colors.end(), [](std::tuple<double, sf::Color> a, std::tuple<double, sf::Color> b) -> bool { return std::get<0>(a) < std::get<0>(b); });
#if DEBUG_COLORS
    for (size_t i = 0;i < colors.size();i++)
    {
        std::cout << std::get<0>(colors[i]) << " & " << std::get<1>(colors[i]) << ",";
    }
    std::cout << std::endl;
#endif

}

double scale(double val, std::tuple<double, double> old_range, std::tuple<double, double> new_range)
{
    double o_max = std::get<1>(old_range);
    double n_max = std::get<1>(new_range);
    double o_min = std::get<0>(old_range);
    double n_min = std::get<0>(new_range);

    return (val - o_min) / (o_max - o_min) * (n_max - n_min) + n_min;
}

sf::Color ColorMap::calculateColor(double val)
{
    if (val <= std::get<0>(colors[0]))
        return std::get<1>(colors[0]);

    if (val >= std::get<0>(colors[colors.size() - 1]))
        return std::get<1>(colors[colors.size() - 1]);

    for (size_t i = 1; i < colors.size(); i++)
    {
        if (std::get<0>(colors[i]) > val)
        {
            sf::Color col(0, 0, 0);
            col.r = (int)scale(val,
                               std::tuple<double, double>(std::get<0>(colors[i]), std::get<0>(colors[i - 1])),
                               std::tuple<double, double>(std::get<1>(colors[i]).r, std::get<1>(colors[i - 1]).r));
            col.g = (int)scale(val,
                               std::tuple<double, double>(std::get<0>(colors[i]), std::get<0>(colors[i - 1])),
                               std::tuple<double, double>(std::get<1>(colors[i]).g, std::get<1>(colors[i - 1]).g));
            col.b = (int)scale(val,
                               std::tuple<double, double>(std::get<0>(colors[i]), std::get<0>(colors[i - 1])),
                               std::tuple<double, double>(std::get<1>(colors[i]).b, std::get<1>(colors[i - 1]).b));
            return col;
        }
    }
}
