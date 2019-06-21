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
    Flock flock;
		float bodiesSize;
		vector<sf::CircleShape> shapes;
		vector<sf::RectangleShape> lines;
		vector<wvu_swarm_std_msgs::vicon_point> obstacles;
		wvu_swarm_std_msgs::vicon_points targets;
		wvu_swarm_std_msgs::flows flows;
		vector<sf::CircleShape> obs_shapes;
		bool game;
		string winner;

    //subscriber input handleing
    void vectorCallback(const wvu_swarm_std_msgs::robot_command_array &msg); //processes info from final_execute
    void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg);   //processes info from virtual_obstacles
    void targetCallback(const wvu_swarm_std_msgs::vicon_points &msg);//processes info from virtual_targets
    void flowCallback(const wvu_swarm_std_msgs::flows &msg);         //processes info from virtual_flows

    //Handle graphics rendering and mouse/key inputs
    PrevIteration HandleInput(PrevIteration _pI);
    void Render();
    bool pause(bool _key_pressed, bool _pause_pressed, bool _pause_sim, sf::RenderWindow* win, sf::Event _event);

    //Specific objects to be rendered.
    void drawObstacles();
    void drawTargets();
    void updateTargetPos();
    void drawFlows();
    void drawGoals();
    void winCheck(int i);

public:
    Sim();
    void Run(ros::NodeHandle _n);
};

#endif
