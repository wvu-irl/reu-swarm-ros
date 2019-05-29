/**
 * 
 *  Swarm simulation
 * 
 *      Simulator uses SFML to simulate swarm behavior
 * 
 *  Henry Vos
 * 
 * 
 *  Compile with
 *      g++ SwarmSim.cpp -oSwarmSim -lsfml-graphics -lsfml-window -lsfml-system
 */

#include <SFML/Graphics.hpp>
#include "GameClock.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "Robot.hpp"

void gameclock::tick()
{
    if (swarm.size() >= 0)
    {
        for (size_t i = 0; i < swarm.size(); i++)
        {
            swarm.at(i).tick();
        }
    }
}

void gameclock::render(sf::RenderWindow *window)
{
    if (swarm.size() >= 0)
    {
        for (size_t i = 0; i < swarm.size(); i++)
        {
            swarm.at(i).render(window);
        }
    }
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