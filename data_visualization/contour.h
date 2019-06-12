#ifndef CONTOUR_H
#define CONTOUR_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>

#define LINE_CHECK_DIST 1

class Vector3D
{
public:
    double x, y, z;

    Vector3D(double x, double y, double z);

    double magnitude() const;

    double dot(const Vector3D &rhs) const;

    Vector3D operator-(const Vector3D &rhs) const
    {
        return Vector3D(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    Vector3D operator*(double rhs) const
    {
        return Vector3D(rhs * x, rhs * y, rhs * z);
    }

    friend std::ostream &operator<<(std::ostream &, const Vector3D &);
};

class ContourMap
{
private:
    sf::Uint8 *cols;
    sf::Image img;
    sf::Texture tex;
    sf::Sprite sprite;

    std::function<double(double, double)> zfunc;

public:
    sf::Rect<int> bounds;
    std::vector<double> levels;

    ContourMap(sf::Rect<int> bounds);

    void render(sf::RenderWindow *window);

    void resemble(std::function<double(double, double)> z);

    void scale(float sx, float sy);

    ~ContourMap();
};
#include "contour.cpp"
#endif