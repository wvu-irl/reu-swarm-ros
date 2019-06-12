#include <iostream>
#include <sstream>
#include <math.h>
#include <thread>
#include "contour.h"

#define DEBUG_CONT_SRC 1

Vector3D::Vector3D(double x, double y, double z) : x(x), y(y), z(z)
{
}

double Vector3D::magnitude() const
{
    return sqrt(x * x + y * y + z * z);
}

double Vector3D::dot(const Vector3D &rhs) const
{
    return x * rhs.x + y * rhs.y + z * rhs.z;
}

std::ostream &operator<<(std::ostream &os, const Vector3D &f)
{
    std::stringstream ss;
    ss << "(" << f.x << ", " << f.y << ", " << f.z << ")";
    return os << ss.str();
}

// Contour map definiton

ContourMap::ContourMap(sf::Rect<int> _bounds)
{
    bounds = sf::Rect<int>(_bounds);
    //this->data = new Raster2D<double>(bounds.height, bounds.width, 0.0);

    cols = (sf::Uint8 *)malloc(bounds.width * bounds.height * 4);
    img.create(bounds.width, bounds.height, cols);

    sprite.scale(sf::Vector2f(2.5, 2.5));
}

ContourMap::~ContourMap()
{
    free(cols);
}

sf::Vector2f intersectPoint(Vector3D _ray_vector, Vector3D _start_point, double level)
{
    Vector3D vect = _start_point - _ray_vector * (((_ray_vector - Vector3D(0.0, 0.0, level)).dot(Vector3D(0.0, 0.0, 1.0))) / (_ray_vector.dot(Vector3D(0.0, 0.0, 1.0))));
    return sf::Vector2f(vect.x, vect.y);
}

void ContourMap::resemble(std::function<double(double, double)> z)
{
#if DEBUG_CONT_SRC
    std::cout << "setting funk" << std::endl;
#endif
    this->zfunc = z;
}

void ContourMap::render(sf::RenderWindow *window)
{
    if (levels.size() > 0)
    {
        for (size_t i = 0; i < bounds.height; i++)
        {
            for (double j = 0; j < bounds.width; j++)
            {
                for (double k = 0; k < levels.size(); k++)
                {
                    Vector3D zc = Vector3D(j, i, zfunc(j, i));
                    Vector3D zu = Vector3D(j, i - 1.5, zfunc(j, i - 1.5));
                    Vector3D zd = Vector3D(j, i - 1.5, zfunc(j, i - 1.5));
                    Vector3D zl = Vector3D(j - 1.5, i, zfunc(j - 1.5, i));
                    Vector3D zr = Vector3D(j + 1.5, i, zfunc(j + 1.5, i));
                    img.setPixel(j, i, sf::Color::Transparent);
                    if (levels.at(k) < zc.z && (levels.at(k) > zu.z || levels.at(k) > zl.z ||
                                                levels.at(k) > zd.z || levels.at(k) > zr.z) ||
                        abs(levels.at(k) - zc.z) < 0.001)
                    {
                        img.setPixel(j, i, sf::Color::White);
                        break;
                    }
                }
            }
        }

        tex.loadFromImage(img);
        sprite.setTexture(tex, true);
        window->draw(sprite);
    }
}