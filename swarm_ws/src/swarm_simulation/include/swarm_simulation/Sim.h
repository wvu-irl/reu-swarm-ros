#include <iostream>
#include "Flock.h"
#include "Body.h"
#include "Pvector.h"
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
#include <wvu_swarm_std_msgs/flows.h>
#include <wvu_swarm_std_msgs/virtual_objects.h>
#include <wvu_swarm_std_msgs/chargers.h>
//#include <swarm_server/robot_id.h>

#ifndef SIM_H
#define SIM_H

// Sim handles the instantiation of a flock of bodies, Sim input, asks the
// model to compute the next step in the stimulation, and handles all of the
// program's interaction with SFML.

struct PrevIteration //struct created solely for click and drag of bots, targets, and obstacles.
{
	bool dragging;
	int botId;
	bool prevClick;
	bool bot; //is true if bot selected.
	bool target; //is true if target selected.
	bool obs; //is true if obstacle selected.
};

class Sim
{

private:
	int NUMBOTS = 1; //number of bots in the simulation; the WHOLE simulation (hub & alice), not just this file.
	sf::RenderWindow window;
	int window_width;
	int window_height;
	bool pauseSim;
	Flock flock;
	vector<sf::CircleShape> shapes;
	vector<sf::RectangleShape> lines;
	vector<sf::Text> texts;
	wvu_swarm_std_msgs::vicon_point num_bots;
	wvu_swarm_std_msgs::vicon_points obstacles;
	wvu_swarm_std_msgs::vicon_points targets;
	wvu_swarm_std_msgs::chargers chargers;
	wvu_swarm_std_msgs::flows flows;
	vector<sf::CircleShape> obs_shapes;
	bool game;
	string winner;


	//text related stuff
	void addText();
	float bodiesSize;

	//Handle graphics rendering and mouse/key inputs
	PrevIteration HandleInput(PrevIteration _pI);
	void Render();
	void clickNdragBots(PrevIteration *_pI, float _mX, float _mY, sf::Event _event);
	void clickNdragTarget(PrevIteration *_pI,float _mX, float _mY, sf::Event _event);
	void clickNdragObstacles(PrevIteration *_pI, float _mX, float _mY, sf::Event _event);
	bool pause(bool _key_pressed, bool _pause_pressed, bool &_pause_sim, sf::RenderWindow* win, sf::Event _event);

	//subscriber input handleing
	void vectorCallback(const wvu_swarm_std_msgs::robot_command_array &msg); //processes info from final_execute
	void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg);   //processes info from virtual_obstacles
	void targetCallback(const wvu_swarm_std_msgs::vicon_points &msg);   //processes info from virtual_targets
	void flowCallback(const wvu_swarm_std_msgs::flows &msg);         //processes info from virtual_flows
	void chargerCallback(const wvu_swarm_std_msgs::chargers &msg);   //processes info from chargers
	void numBotsCallback(const wvu_swarm_std_msgs::vicon_point &msg); //gets number of bots in the sim.

	//Specific objects to be rendered.
	void drawObstacles();
	void drawTargets();
	void updateTargetPos();
	void drawFlows();
	void drawChargers(); //draws chargers

public:
	Sim();
	void Run(ros::NodeHandle _n);
};

#endif
