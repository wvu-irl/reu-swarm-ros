#include <iostream>
#include <swarm_simulation/Flock.h>
#include <swarm_simulation/Body.h>
#include <swarm_simulation/Pvector.h>
#include <swarm_simulation/Sim.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <math.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
//#include "ros/ros.h"

void Sim::vectorCallback(const wvu_swarm_std_msgs::robot_command_array &msg)
{
	for (int i = 0; i < msg.commands.size(); i++)
	{
		for (int j = 0; j < flock.flock.size(); j++)
		{
			if (msg.commands.at(i).rid == j)
			{
				flock.flock.at(j).velocity.set(
						0.1 * msg.commands.at(i).r
								* cos(flock.flock.at(j).angle(flock.flock.at(j).velocity) + M_PI / 180.0 * msg.commands.at(i).theta),
						0.1 * msg.commands.at(i).r
								* sin(flock.flock.at(j).angle(flock.flock.at(j).velocity) + M_PI / 180.0 * msg.commands.at(i).theta));
			}
		}
	}
}

// Construct window using SFML
Sim::Sim()
{
	this->bodiesSize = 12.0;
	sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	this->window_height = 600;
	this->window_width = 300;

	//std::cout<<"Window h;w = "<<window_height << "; "<<window_width<<"\n";
	this->window.create(sf::VideoMode(window_width, window_height, desktop.bitsPerPixel),
			"Flocking Simulation Aleks Hatfield", sf::Style::None);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void Sim::Run(ros::NodeHandle _n)
{
	char letters[100] = { 'D', 'E', 'P', 'A', 'N', 'J', 'G', 'A', 'C', 'T', 'M', 'A', 'M', 'D', 'S', 'C', 'N', 'H', 'V',
			'A', 'N', 'Y', 'N', 'C', 'R', 'I', 'V', 'T', 'K', 'Y', 'T', 'N', 'O', 'H', 'L', 'A', 'I', 'N', 'M', 'S', 'I', 'L',
			'A', 'L', 'M', 'E', 'M', 'O', 'A', 'R', 'M', 'I', 'F', 'L', 'T', 'X', 'I', 'A', 'W', 'I', 'C', 'A', 'M', 'N', 'O',
			'R', 'K', 'A', 'W', 'V', 'N', 'V', 'N', 'E', 'C', 'O', 'N', 'D', 'S', 'D', 'M', 'T', 'W', 'A', 'I', 'D', 'W', 'Y',
			'U', 'T', 'O', 'K', 'N', 'M', 'A', 'Z', 'A', 'K', 'H', 'I' };

	for (int i = 0; i < 50; i++)
	{
		char temp[2] = { letters[2 * i], letters[2 * i + 1] };
		Body b(25*(int)(i /10), 50*(i % 10), temp); // Starts all bodies in the center of the screen
		sf::CircleShape shape(8, 3);

		// Changing the Visual Properties of the shape.
		shape.setPosition(b.location.x, b.location.y); // Sets position of shape to random location that body was set to.
		shape.setOrigin(12,12);
		//shape.setPosition(window_width, window_height); // Testing purposes, starts all shapes in the center of screen.
		shape.setFillColor(sf::Color::Yellow);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1);
		shape.setRadius(bodiesSize);

		// Adding the body to the flock and adding the shapes to the vector<sf::CircleShape>
		flock.addBody(b);
		shapes.push_back(shape);
		window.draw(shape);
	}
	window.display();
	sleep(1);
	ros::Publisher pub = _n.advertise < wvu_swarm_std_msgs::vicon_bot_array > ("vicon_array", 1000); //Publishes like Vicon
	ros::Subscriber sub = _n.subscribe("final_execute", 1000, &Sim::vectorCallback, this); //subscribes to funnel
	ros::Rate loop_rate(10);

	//publishes initial information for each bot
	wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();
	pub.publish(vb_array);
	ros::spinOnce();
	while (window.isOpen() && ros::ok())
	{
		//HandleInput();
		Render();
		wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();

		//publishing vicon_bot_array
		//flock.printMessage(vb_array);
		pub.publish(vb_array);
		ros::spinOnce();

	}
}

void Sim::HandleInput()
{
	sf::Event event;
	while (window.pollEvent(event))
	{
		// "close requested" event: we close the window
		// Implemented alternate ways to close the window. (Pressing the escape, X, and BackSpace key also close the program.)
		if ((event.type == sf::Event::Closed)
				|| (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
				|| (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::BackSpace)
				|| (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::X))
		{
			window.close();
		}
	}

}

void Sim::Render()
{
	window.clear();

	// Draws all of the bodies out, and applies functions that are needed to update.
	for (int i = 0; i < shapes.size(); i++)
	{
		// Matches up the location of the shape to the body
		shapes[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);

		// Calculates the angle where the velocity is pointing so that the triangle turns towards it.
		float theta = flock.getBody(i).angle(flock.getBody(i).velocity);
		shapes[i].setRotation(theta);

		// Prevent bodies from moving off the screen through wrapping
		// If body exits right boundary
		if (shapes[i].getPosition().x > window_width)
			shapes[i].setPosition(shapes[i].getPosition().x - window_width, shapes[i].getPosition().y);
		// If body exits bottom boundary
		if (shapes[i].getPosition().y > window_height)
			shapes[i].setPosition(shapes[i].getPosition().x, shapes[i].getPosition().y - window_height);
		// If body exits left boundary
		if (shapes[i].getPosition().x < 0)
			shapes[i].setPosition(shapes[i].getPosition().x + window_width, shapes[i].getPosition().y);
		// If body exits top boundary
		if (shapes[i].getPosition().y < 0)
			shapes[i].setPosition(shapes[i].getPosition().x, shapes[i].getPosition().y + window_height);
		window.draw(shapes[i]);
	}
	flock.flocking();
	// Applies the three rules to each body in the flock and changes them accordingly.

	window.display();

}

