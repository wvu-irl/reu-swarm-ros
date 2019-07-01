#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <math.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <swarm_simulation/Font.h>
#include <visualization/alice_pov.h>

//#include "ros/ros.h"
void AlicePOV::mapCallback(const wvu_swarm_std_msgs::alice_mail_array &msg)
{
	map = msg;
}

// Construct window using SFML
AlicePOV::AlicePOV(void)
{
	bodiesSize = 7.5;
	sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	window_height = 600;
	window_width = 600;
	this->window.create(sf::VideoMode(window_width, window_height, desktop.bitsPerPixel), "Alice POV",
			sf::Style::Titlebar);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void AlicePOV::Run(ros::NodeHandle _n)
{
	//Initializes all publishers and subscribers

	ros::Subscriber sub = _n.subscribe("alice_mail_array", 1000, &AlicePOV::mapCallback, this); //subscribes to funnel

	ros::Rate loopRate(15);

	while (window.isOpen() && ros::ok()) //main while loop (runs the simulation).
	{
		HandleInput();
		Render();
		ros::spinOnce();
		loopRate.sleep();
	}
}
void AlicePOV::HandleInput() //switches the robot viewed depending on what buttons are pressed
{
	sf::Event event;
	while (window.pollEvent(event))
	{
		int i = 0; //iterator for dragging while loop
		float mX = event.mouseButton.x; //mouse x pos
		float mY = event.mouseButton.y; //mouse y pos

		//---------- Pressing the escape key will close the program
		if ((event.type == sf::Event::Closed)
				|| (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
		{
			window.close();
		}

		if (event.type == sf::Event::TextEntered)
		{
			if (event.text.unicode < 128)
				name = (int) (static_cast<char>(event.text.unicode) - '0');
			std::cout << name << std::endl;
		}

	}
}

void AlicePOV::Render() //draws changes in simulation states to the window.
{
	window.clear();

	for (int i = 0; i < map.mails.size(); i++)
	{

		if (map.mails.at(i).name == name)
		{
			sf::CircleShape shape(0);
			// Changing the Visual Properties of the robot
			shape.setPosition(300, 300); // Sets position of shape to random location that body was set to.
			shape.setOrigin(bodiesSize, bodiesSize);
			shape.setFillColor(
					sf::Color(255 - (int) map.mails.at(name).contourVal, (int) map.mails.at(name).contourVal, 0, 255));
			shape.setOutlineColor(sf::Color::Green);
			shape.setOutlineThickness(1);
			shape.setRadius(bodiesSize);
			window.draw(shape);

			for (int j = 0; j < map.mails.at(i).neighborMail.size(); j++)
			{
				wvu_swarm_std_msgs::neighbor_mail temp = map.mails.at(i).neighborMail.at(j);
				sf::CircleShape shape(0);
				// Changing the Visual Properties of the robot
				shape.setPosition(300+ 3*temp.x, 300-3*temp.y); // Sets position of shape to random location that body was set to.
				shape.setOrigin(bodiesSize, bodiesSize);
				shape.setFillColor(
						sf::Color(255 - (int) map.mails.at(temp.name).contourVal,0,  (int) map.mails.at(temp.name	).contourVal, 255));
				shape.setOutlineColor(sf::Color::White);
				shape.setOutlineThickness(1);
				shape.setRadius(bodiesSize);
				window.draw(shape);
			}
			for (int j = 0; j < map.mails.at(i).targetMail.size(); j++)
						{
							wvu_swarm_std_msgs::obj_mail temp = map.mails.at(i).targetMail.at(j);
							sf::CircleShape shape(0);
							// Changing the Visual Properties of the obstacle
							shape.setPosition(300+ 3*temp.x, 300-3*temp.y); // Sets position of shape to random location that body was set to.
							shape.setOrigin(bodiesSize, bodiesSize);
							shape.setFillColor(sf::Color::Green);
							shape.setRadius(bodiesSize);
							window.draw(shape);
						}
		}
	}

	window.display(); //updates display
}
//void Sim::addText() //adds text for the state abbreviations
//{
//
//	sf::Font font;
//	font.loadFromMemory(&ComicSansMS3_ttf, ComicSansMS3_ttf_len);
//
//	for (int i = 0; i < shapes.size(); i++)
//	{
//
//		//creates text on the bodies
//
//		sf::Text text;
//		text.setFont(font);
//		text.setCharacterSize(10);
//		text.setColor(sf::Color::Red);
//
//		std::string temp(flock.getBody(i).id);
//		text.setString(temp.substr(0, 2));
//		text.setStyle(sf::Text::Bold);
//		text.setOrigin(7.5, 7.5);
//
//		texts.push_back(text);
//
//		window.draw(text);
//
//		window.draw(texts[i]);
//
//		texts[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);
//	}
//}

