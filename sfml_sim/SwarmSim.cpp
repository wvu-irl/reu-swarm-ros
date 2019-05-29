#include "SimBase.hpp"

/**
 * Function that produces ideal vectors
 */
sf::Vector2f ruleCallback(struct model m)
{
    return sf::Vector2f(0, 0);
}

int main()
{
    puts("starting");
    sf::ContextSettings sett;
    sett.antialiasingLevel = 3;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Swarm Simulation", sf::Style::Default, sett);

    // setup
    for (size_t i = 0; i < 50; i++)
    {
        swarm.push_back(HolonomicBot(20 + (robot_size + 3) * (int)(i % 20), 20 + (robot_size + 3) * (int)(i / 20), ruleCallback));
    }

    // end setup
    gameclock::init(&window, WIDTH, HEIGHT, 100.0f, 60.0f);
    return 0;
}