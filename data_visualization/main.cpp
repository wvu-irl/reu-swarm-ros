/*
 *  Compile using
 * 
 *      g++ main.cpp -oContour.o -lsfml-graphics -lsfml-window -lsfml-system
 */

#include "GameClock.hpp"
#include "contour.h"

#include <math.h>

#define WIDTH 600
#define HEIGHT 600

ContourMap *cont;

static int tick = 0;

double function(double x, double y)
{
    x -= 100;
    y -= 100;
    x /= 10;
    y /= 10;
    return x * x - y * y;
}

void incrTick()
{
    tick++;
    tick %= 100;
}

void gameclock::tick()
{
    incrTick();
}

void gameclock::render(sf::RenderWindow *window)
{
    cont->render(window);
}

void gameclock::buttonAction(int id)
{
}

void gameclock::keyPress(sf::Keyboard::Key key)
{
}

void gameclock::keyRelease(sf::Keyboard::Key key)
{
}

int main(int argc, char **argv)
{
    cont = new ContourMap(sf::Rect<int>(10, 10, 200, 200));

    for (double i = -10.0; i <= 10.0; i += 20.0 / 5.0)
    {
        cont->levels.push_back(i);
        std::cout << "Added level: " << i << std::endl;
    }
    cont->resemble(function);

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Contour Plotting", sf::Style::Default);

    gameclock::init(&window, WIDTH, HEIGHT, 100.0f, 60.0f);
}