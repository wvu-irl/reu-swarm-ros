#include <iostream>
#include "Flock.h"
#include "Body.h"
#include "Pvector.h"
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
//#include <swarm_server/robot_id.h>

#ifndef SIM_H
#define SIM_H

// Sim handles the instantiation of a flock of bodies, Sim input, asks the
// model to compute the next step in the stimulation, and handles all of the
// program's interaction with SFML.

class Sim {
private:
    sf::RenderWindow window;
    int window_width;
    int window_height;

    Flock flock;
    float bodiesSize;
    vector<sf::CircleShape> shapes;

    void Render();
    void HandleInput();

public:
    Sim();
    void Run(ros::NodeHandle _n);
};

#endif
