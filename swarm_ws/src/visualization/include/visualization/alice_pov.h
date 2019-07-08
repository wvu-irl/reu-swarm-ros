#include <iostream>
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include <ros/ros.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <wvu_swarm_std_msgs/map.h>
#include <swarm_simulation/Font.h>
#ifndef POV_H
#define POV_H

class AlicePOV
{
private:
	sf::RenderWindow window;
	sf::RenderWindow window2;
	int window_width;
	int window_height;

	wvu_swarm_std_msgs::alice_mail_array mail;
	wvu_swarm_std_msgs::map map;
	int name;
//	vector<sf::CircleShape> neighbors;
//	vector<sf::CircleShape> neighbors;
//	vector<sf::RectangleShape> lines; //for flows
//	vector<sf::Text> texts;
//	vector<sf::CircleShape> obs_shapes; //for new obstacles


	//text related stuff
	//void addText();
	float bodiesSize;

	//subscriber input handleing
	void mailCallback(const wvu_swarm_std_msgs::alice_mail_array &msg);
	void mapCallback(const wvu_swarm_std_msgs::map &msg);
	void HandleInput();
	void drawMail();
	void drawMsg(ros::ServiceClient _client);
public:
	AlicePOV();
	void Run(ros::NodeHandle _n);
	void Render(ros::ServiceClient _client);

};

#endif
