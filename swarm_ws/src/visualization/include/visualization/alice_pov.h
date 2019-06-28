#include <iostream>
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <swarm_simulation/Font.h>
#ifndef POV_H
#define POV_H

class AlicePOV
{
private:
	sf::RenderWindow window;
	int window_width;
	int window_height;

	wvu_swarm_std_msgs::alice_mail_array map;

//	vector<sf::CircleShape> neighbors;
//	vector<sf::CircleShape> neighbors;
//	vector<sf::RectangleShape> lines; //for flows
//	vector<sf::Text> texts;
//	vector<sf::CircleShape> obs_shapes; //for new obstacles


	//text related stuff
	//void addText();
	float bodiesSize;

	//subscriber input handleing
	void mapCallback(const wvu_swarm_std_msgs::alice_mail_array &msg);

public:
	AlicePOV(void);
	void Run(ros::NodeHandle _n);
	void Render();

};

#endif
