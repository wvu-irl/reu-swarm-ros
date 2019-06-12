/*
 *  Compile using
 * 
 *      g++ main.cpp -oContour.o -lsfml-graphics -lsfml-window -lsfml-system
 */

#include "contour.h"

#include <math.h>

#define WIDTH 600
#define HEIGHT 600

ContourMap *cont;

static int g_tick = 0;

double function(double x, double y)
{
    x -= 250;
    y -= 250;
    x /= 50;
    y /= 50;
    return (x * x - y * y) * sin(M_PI / 50 * g_tick);
}

void tick()
{
    std::cout << "Tick" << std::endl;

    g_tick++;
    g_tick %= 100;
}

void render(sf::RenderWindow *window)
{
    cont->render(window);
}

int main(int argc, char **argv)
{
    cont = new ContourMap(sf::Rect<int>(10, 10, 500, 500));

    for (double i = -20.0; i <= 20.0; i += 40.0 / 16.0)
    {
        cont->levels.push_back(i);
        std::cout << "Added level: " << i << std::endl;
    }
    cont->resemble(function);

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Contour Plotting", sf::Style::Default);

    // gameclock::init(&window, WIDTH, HEIGHT, 100.0f, 30.0f);
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            switch (event.type)
            {
            case sf::Event::Closed:
                window.close();
                break;
            }
        }

        tick();

        if (window.isOpen())
        {
            window.clear();
            render(&window);
            window.display();
        }
    }
}