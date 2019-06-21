#include <iostream>
#include "Flock.h"
#include "Body.h"
#include "Pvector.h"
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
#include <wvu_swarm_std_msgs/flows.h>
//#include <swarm_server/robot_id.h>

#ifndef SIM_H
#define SIM_H

// Sim handles the instantiation of a flock of bodies, Sim input, asks the
// model to compute the next step in the stimulation, and handles all of the
// program's interaction with SFML.

struct PrevIteration{

	bool dragging;
	int botId;
	bool prevClick;
};

class Sim {
private:
    sf::RenderWindow window;
    int window_width;
    int window_height;
    void vectorCallback(const wvu_swarm_std_msgs::robot_command_array &msg);
    void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg);
    void targetCallback(const wvu_swarm_std_msgs::vicon_points &msg);
    void flowCallback(const wvu_swarm_std_msgs::flows &msg);
    void drawObstacles();
    void drawTargets();
    void updateTargetPos();
    void drawFlows();
    Flock flock;
    float bodiesSize;
    vector<sf::CircleShape> shapes;
    vector<sf::RectangleShape> lines;
    //vector<sf::Text> texts;
    vector<wvu_swarm_std_msgs::vicon_point> obstacles;
    wvu_swarm_std_msgs::vicon_points targets;
    wvu_swarm_std_msgs::flows flows;
    vector<sf::CircleShape> obs_shapes;
    void Render();
    PrevIteration HandleInput(PrevIteration _pI);
    bool pause(bool _key_pressed, bool _pause_pressed, bool _pause_sim, sf::RenderWindow* win, sf::Event _event);

public:
    Sim();
    void Run(ros::NodeHandle _n);
};

#endif
