#include "ColorMap.h"
ColorMap::ColorMap(std::pair<double, sf::Color> min, std::pair<double, sf::Color> max)
{
    colors.push_back(min);
    colors.push_back(max);
    gradiant_ = false;
}

void ColorMap::addColor(std::pair<double, sf::Color> val)
{
    int l = 0;
    int r = colors.size() - 1;
    int mid = (r + l) / 2;

    bool found = false;

    while (l < r && !found)
    {
        mid = (r + l) / 2;
        if (val.first < colors[mid].first)
        {
            r = mid - 1;
        }
        else if (val.first > colors[mid].first)
        {
            l = mid + 1;
        }
        else
        {
            found = true;
        }
    }
    if (!found)
    {
        if (colors[mid].first < val.first)
            colors.insert(colors.begin() + mid + 1, val);
        else
            colors.insert(colors.begin() + mid, val);
    }
    else
    {
        colors[mid] = val;
    }
}

double scale(double val, std::tuple<double, double> old_range, std::tuple<double, double> new_range)
{
    double o_max = std::get<0>(old_range) > std::get<1>(old_range) ? std::get<0>(old_range) : std::get<1>(old_range);
    double n_max = std::get<0>(new_range) > std::get<1>(new_range) ? std::get<0>(new_range) : std::get<1>(new_range);
    double o_min = std::get<0>(old_range) < std::get<1>(old_range) ? std::get<0>(old_range) : std::get<1>(old_range);
    double n_min = std::get<0>(new_range) < std::get<1>(new_range) ? std::get<0>(new_range) : std::get<1>(new_range);

    return (val - o_min) / (o_max - o_min) * (n_max - n_min) + n_min;
}

sf::Color ColorMap::calculateColor(double val)
{
    int l = 0;
    int r = colors.size() - 1;
    int mid = (r + l) / 2;

    bool found = false;

    while (l < r && !found)
    {
        mid = (r + l) / 2;
        if (val < colors[mid].first)
        {
            r = mid - 1;
        }
        else if (val > colors[mid].first)
        {
            l = mid + 1;
        }
        else
        {
            found = true;
        }
    }
    if (found)
        return colors[mid].second;

    if (gradiant_)
    {
        if (colors[mid].first > val && mid > 0)
            mid--;

        if (mid <= 0)
            return colors[0].second;
        else if (mid >= colors.size() - 1)
            return colors[colors.size() - 1].second;

        sf::Color grad(0, 0, 0);
        grad.r = scale(val, std::tuple<double, double>(colors[mid].first, colors[mid + 1].first), std::tuple<double, double>(colors[mid].second.r, colors[mid + 1].second.r));
        grad.g = scale(val, std::tuple<double, double>(colors[mid].first, colors[mid + 1].first), std::tuple<double, double>(colors[mid].second.g, colors[mid + 1].second.g));
        grad.b = scale(val, std::tuple<double, double>(colors[mid].first, colors[mid + 1].first), std::tuple<double, double>(colors[mid].second.b, colors[mid + 1].second.b));

        return grad;
    }
    else
    {
        if (colors[mid].first > val)
            return colors[mid - (mid > 0 ? 1 : 0)].second;
        else
            return colors[mid].second;
    }
}

void ColorMap::isGradiant(bool is)
{
    gradiant_ = is;
}

bool ColorMap::isGradiant() const
{
    return gradiant_;
}