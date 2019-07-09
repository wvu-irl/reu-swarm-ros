#ifndef CONTOUR_CPP
#define CONTOUR_CPP
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <string.h>
#include <thread>
#include <future>
#include <fstream>
#include <iomanip>

#include "calculation.cu"

#include <visualization/contour.h>
#include <contour_node/level_description.h>

#define DEBUG_CONT_SRC 0

#define USE_CPU 0

// constructor
Vector3D::Vector3D(double x, double y, double z) :
		x(x), y(y), z(z)
{
}

// calculates the magnitude
double Vector3D::magnitude() const
{
	return sqrt(x * x + y * y + z * z);
}

// does the dot product of this and another vector
double Vector3D::dot(const Vector3D &rhs) const
{
	return x * rhs.x + y * rhs.y + z * rhs.z;
}

// cout operator
std::ostream &operator<<(std::ostream &os, const Vector3D &f)
{
	std::stringstream ss;
	ss << "(" << f.x << ", " << f.y << ", " << f.z << ")";
	return os << ss.str();
}

// Contour map definiton
// constructor
ContourMap::ContourMap(sf::Rect<int> _bounds, ColorMap cm) :
		color_mapping(cm) // setting the color map
{
#if DEBUG_CONT_SRC
    std::cout << "\033[32mCreating contour map\033[0m" << std::endl;
#endif
	bounds = sf::Rect<int>(_bounds); // setting the bounds
	sprite.setPosition(bounds.left, bounds.top); // sets the sprite position
	cols = (sf::Uint8 *)malloc(sizeof(sf::Uint8) * bounds.width * bounds.height * 4);
}

void ContourMap::scale(float sx, float sy)
{
	this->sprite.scale(sf::Vector2f(sx, sy)); // scales the sprite
}

// destructor
ContourMap::~ContourMap()
{
	hyperthread::destruct();
	free(cols);
}

// setter for the function
void ContourMap::resemble(wvu_swarm_std_msgs::map_level z)
{
#if DEBUG_CONT_SRC
    std::cout << "setting funk" << std::endl;
#endif
	this->zfunc = z;
}

void ContourMap::tick()
{
	hyperthread::color *colors = (hyperthread::color *)malloc(sizeof(hyperthread::color) * color_mapping.colors.size());
	double *c_levels = (double *)malloc(sizeof(double) * color_mapping.colors.size());

#if !USE_CPU

	for (size_t i = 0;i < color_mapping.colors.size();i++)
	{
		sf::Color color = std::get<1>(color_mapping.colors[i]);
		colors[i] = (hyperthread::color){color.r, color.g, color.b};
		c_levels[i] = std::get<0>(color_mapping.colors[i]);
	}

	hyperthread::calc(cols, 
		levels,
		bounds.width, bounds.height,
		colors, c_levels, color_mapping.colors.size(), zfunc);
#if DEBUG_CONT_SRC
	 for (size_t i = 0;i < bounds.width * bounds.height * 4;i++)
	 {
	 	std::cout << "COL[" << i << "]: R=" << (int)cols[i] << ", G=" << (int)cols[i + 1] << ", B=" << (int)cols[i + 2] << std::endl;
	 }
#endif
#endif

	img.create(bounds.width, bounds.height, cols);
#if USE_CPU
#if DEBUG_CONT_SRC
	std::cout << "\033[1mStarting calc\033[0m" << std::endl;
#endif
	for (size_t idx = 0;idx < bounds.width * bounds.height;idx++)
	{
#if DEBUG_CONT_SRC
				std::cout << " \033[33mStarting point calc\033[0m" << std::endl;
#endif
		double x = (double)((int)idx % (int)bounds.width);
		double y = (double)((int)idx / (int)bounds.width);
		x -= 640;
		y -= 400;
		x *= 200.0 / (double)bounds.width;
		y *= 100.0 / (double)bounds.height;
#if DEBUG_CONT_SRC
				std::cout << " \033[33mCompleted shift\033[0m" << std::endl;
#endif

		double *z = (double *)malloc(sizeof(double));
		*z = 0;
		double theta = x == 0 ? (y > 0 ? M_PI_2 : -M_PI_2) : (atan(y/x) + (y < 0 ? M_PI : 0));
		double r = sqrt(x*x + y*y);

		for (size_t i = 0;i < zfunc.functions.size();i++)
		{
#if DEBUG_CONT_SRC
				std::cout << "  \033[32mCalculating for : " << i << "\033[0m" << std::endl;
//				std::cout << "   \033[34mGaussian:\n" << zfunc.functions[i] << "\033[0m" << std::endl;
#endif
			wvu_swarm_std_msgs::gaussian curr_eq = zfunc.functions[i];
			double a = curr_eq.ellipse.x_rad;
			double b = curr_eq.ellipse.y_rad;

			double x_app = r * cos(theta + curr_eq.ellipse.theta_offset) - zfunc.functions[i].ellipse.offset_x;
			double y_app = r * sin(theta + curr_eq.ellipse.theta_offset) - zfunc.functions[i].ellipse.offset_y;

			double re = a != 0 && b != 0 ? sqrt(a * a * x_app * x_app + y_app * y_app * b * b) / (a * b) : 0;
#if DEBUG_CONT_SRC
				std::cout << "  \033[32mCompleted transform\033[0m" << std::endl;
#endif
			if (re < M_PI / 1.245)
			{
#if DEBUG_CONT_SRC
				std::cout << "   \033[31mInside ellipse\033[0m" << std::endl;
#endif
				double amp = curr_eq.amplitude / 2.0;
				*z += amp * cos(1.245 * re) + amp;
			}
//				*z += curr_eq.amplitude * pow(M_E, (-re * re) / 2.0);
		}

#if DEBUG_CONT_SRC
//		std::cout << "   For Point: " << x << ", " << y << "\t z = " << *z << std::endl;
//		std::cout << "   Bounds:    " << bounds.width << ", " << bounds.height << std::endl;
//		std::cout << "   Pixel:     " << (int)idx % (int)bounds.width << ", " << (int)idx / (int)bounds.width << std::endl;
		printf("   For point: % 4.2lf, % 4.2lf :: z = % 2.2lf\n   Bounds: %d, %d\n   Pixel: % 4d, % 3d\n",
				x, y, *z, bounds.width, bounds.height, (int)idx % (int)bounds.width, (int)idx / (int)bounds.width);
#endif

		img.setPixel((int)idx % (int)bounds.width, (int)idx / (int)bounds.width, color_mapping.calculateColor(*z));
#if DEBUG_CONT_SRC
		std::cout << "   Set color" << std::endl;
#endif
		free(z);
	}
#endif
	free(colors);
	free(c_levels);
}

void ContourMap::render(sf::RenderTexture *window)
{
	// draws the image to the screen
	tex.loadFromImage(img);
	sprite.setTexture(tex, true);
	window->draw(sprite);
}

// accessors
void ContourMap::setColorMap(ColorMap cm)
{
	color_mapping = cm;
}

ColorMap *ContourMap::getColorMap()
{
	return &color_mapping;
}
#endif
