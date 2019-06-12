#include <iostream>
#include <sstream>
#include <math.h>
#include <thread>
#include <future>
#include "contour.h"

#define DEBUG_CONT_SRC 0

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

    //sprite.scale(sf::Vector2f(2.5, 2.5));
    sprite.setPosition(bounds.left, bounds.top);
}

void ContourMap::scale(float sx, float sy)
{
    this->sprite.scale(sf::Vector2f(sx, sy));
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

Vector3D *drawPix(Vector3D points[5], double level)
{
    std::function<bool(double, double, double)> isBtw([](double l, double r, double mid) -> bool {
        return (l < r ? l : r) <= mid && mid <= (l < r ? r : l);
    });

    Vector3D *draw = NULL;

    for (size_t i = 1; i < 5; i++)
    {
        if (isBtw(points[0].z, points[i].z, level))
        {
            draw = points;
            break;
        }
    }
    return draw;
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
                    Vector3D zu = Vector3D(j, i - LINE_CHECK_DIST, zfunc(j, i - LINE_CHECK_DIST));
                    Vector3D zd = Vector3D(j, i - LINE_CHECK_DIST, zfunc(j, i - LINE_CHECK_DIST));
                    Vector3D zl = Vector3D(j - LINE_CHECK_DIST, i, zfunc(j - LINE_CHECK_DIST, i));
                    Vector3D zr = Vector3D(j + LINE_CHECK_DIST, i, zfunc(j + LINE_CHECK_DIST, i));
                    img.setPixel(j, i, sf::Color::Transparent);
                    Vector3D vects[] = {zc, zu, zd, zl, zr};
                    Vector3D *draw = drawPix(vects, levels.at(k));
                    if (draw != NULL)
                    {
                        img.setPixel((int)draw->x, (int)draw->y, levels.at(k) < 0 ? sf::Color::Red : sf::Color::White);
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