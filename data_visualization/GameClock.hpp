#pragma once
#include <stdlib.h>
#include <SFML/Graphics.hpp>
#include <SFML/System/Time.hpp>
#include <chrono>
#include "MenuControl.hpp"


namespace gameclock
{
size_t width, height;
float scalex = 1, scaley = 1;

sf::Text fpsdisp;
int fps;

void tick();
void render(sf::RenderWindow *window);

void buttonAction(int id);
void keyPress(sf::Keyboard::Key key);
void keyRelease(sf::Keyboard::Key key);

void eventHandeling(sf::Event event, sf::RenderWindow *window)
{
    
    switch (event.type)
    {
    case sf::Event::Closed:
        window->close();
        break;
    case sf::Event::MouseMoved:
        menuGraphics::mouseMove((int)(event.mouseMove.x * scalex), (int)(event.mouseMove.y * scaley));
        break;
    case sf::Event::MouseButtonPressed:
        menuGraphics::mouseClick((int)(event.mouseButton.x * scalex), (int)(event.mouseButton.y * scaley));
        break;
    case sf::Event::MouseButtonReleased:
        buttonAction(menuGraphics::mouseRelease());
        break;
    case sf::Event::Resized:
        scalex = (float)width / (float)event.size.width;
        scaley = (float)height / (float)event.size.height;
        break;
    case sf::Event::KeyPressed:
        keyPress(event.key.code);
        break;
    case sf::Event::KeyReleased:
        keyRelease(event.key.code);
        break;
    }
}
int init(sf::RenderWindow *window, size_t width, size_t height, float fps, float tps)
{
    // clock variables
    auto lastTime = std::chrono::high_resolution_clock::now();
    double ammountOfTicks = (double)tps;
    double ns = 1000000000 / ammountOfTicks;
    double delta = 0;
    long timer = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000000; // in ms
    int frames = 0;
    // end clock variables

    

    //      General
    float x = 0, y = 0;

    while (window->isOpen())
    {
        sf::Event event;
        while (window->pollEvent(event))
        {
            eventHandeling(event, window);
        }

        // Game Clock
        auto now = std::chrono::high_resolution_clock::now();
        long durVal = std::chrono::duration_cast<std::chrono::nanoseconds>((now - lastTime)).count();
        lastTime = now;

        delta += durVal / ns;
        while (delta >= 1)
        {
            tick();
            delta--;
        }

        if (window->isOpen())
        {
            window->clear();
            render(window);
            window->display();
            frames++;
        }

        if (std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000000.0 - (double)timer > 1000)
        {
            timer += 1000;
            fps = frames;
            frames = 0;

            char fpsstr[32];
            sprintf(fpsstr, "%4d FPS", (int)fps);
            fpsdisp.setString(fpsstr);
        }
        sf::sleep(sf::microseconds((int)(1000000.0 / fps))); // clocking fps to reduce processor intensiveness
                                                               // 100 fps
    }

    return 0;
}

} // namespace gameclock