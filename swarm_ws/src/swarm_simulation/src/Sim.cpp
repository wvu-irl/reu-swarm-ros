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
	for (int i = 0; i < flock.flock.size(); i++)
	{
		if (flock.flock.at(i).updatedCommand == false)
		{
			for (int j = 0; j < msg.commands.size(); j++)
			{
				if (msg.commands.at(j).rid == i)
				{
					//std::cout << flock.flock.at(j).angle(flock.flock.at(j).velocity) << std::endl;

//				std::cout<<"--------- Bot ID "<<j<<"--------\n";
//				float o = flock.flock.at(j).angle(flock.flock.at(j).velocity);
//				float n = msg.commands.at(i).theta;
//				std::cout<<"v_old (x,y) = "<<flock.flock.at(j).velocity.x<<","<<flock.flock.at(j).velocity.y<<"\n";
//				std::cout<<"heading: "<<o<<"| theta: "<<n<<"\n";
//				^ will print out fun facts if desired.

//				flock.flock.at(j).velocity.set(
//						msg.commands.at(i).r
//								* cos(flock.flock.at(j).angle(flock.flock.at(j).velocity) - M_PI_2 +  msg.commands.at(i).theta),
//						 msg.commands.at(i).r
//								* sin(flock.flock.at(j).angle(flock.flock.at(j).velocity) - M_PI_2 + msg.commands.at(i).theta));
//       ^the way this code was before, just in case.

//Prevents sudden turns
//				if (msg.commands.at(j).theta > M_PI / 6)
//				{
//					flock.flock.at(i).velocity.set(
//							0.000001 * cos(flock.flock.at(i).angle(flock.flock.at(i).velocity) + M_PI / 18),
//							0.000001 * sin(flock.flock.at(i).angle(flock.flock.at(i).velocity) + M_PI / 18));
//
//				} else if (msg.commands.at(j).theta < -M_PI / 6)
//				{
//					flock.flock.at(i).velocity.set(
//							0.000001 * cos(flock.flock.at(i).angle(flock.flock.at(i).velocity) - M_PI / 18),
//							0.000001 * sin(flock.flock.at(i).angle(flock.flock.at(i).velocity) - M_PI / 18));
//
//				} else
//				{
					flock.flock.at(i).velocity.set(
							1 * msg.commands.at(j).r
									* cos(flock.flock.at(i).angle(flock.flock.at(i).velocity) + msg.commands.at(j).theta),
							1 * msg.commands.at(i).r
									* sin(flock.flock.at(i).angle(flock.flock.at(i).velocity) + msg.commands.at(j).theta));
					//		}
					flock.flock.at(i).updatedCommand = true;

//				std::cout<<"new (sum) angle"<< o +  n<<"\n";
//				std::cout<<"v_new (x,y) = "<<flock.flock.at(j).velocity.x<<","<<flock.flock.at(j).velocity.y<<"\n";
//				std::cout<<"-----------------------------\n";
//				^more fun facts, if ya want um.
				}
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
	char letters[100] =
	{ 'D', 'E', 'P', 'A', 'N', 'J', 'G', 'A', 'C', 'T', 'M', 'A', 'M', 'D', 'S', 'C', 'N', 'H', 'V', 'A', 'N', 'Y', 'N',
			'C', 'R', 'I', 'V', 'T', 'K', 'Y', 'T', 'N', 'O', 'H', 'L', 'A', 'I', 'N', 'M', 'S', 'I', 'L', 'A', 'L', 'M', 'E',
			'M', 'O', 'A', 'R', 'M', 'I', 'F', 'L', 'T', 'X', 'I', 'A', 'W', 'I', 'C', 'A', 'M', 'N', 'O', 'R', 'K', 'A', 'W',
			'V', 'N', 'V', 'N', 'E', 'C', 'O', 'N', 'D', 'S', 'D', 'M', 'T', 'W', 'A', 'I', 'D', 'W', 'Y', 'U', 'T', 'O', 'K',
			'N', 'M', 'A', 'Z', 'A', 'K', 'H', 'I' };

	int x = 50; //x inital positions for the bots.

	for (int i = 0; i < 50; i++)
	{
		char temp[2] = {letters[2 * i], letters[2 * i + 1]};
		int y = 50 * (i%10)+ 50; // y pos of bots

//		std::cout<<"----Bot ID: "<<temp[0]<<temp[1]<<"-------\n";
//		std::cout<<"x,y: "<<x<<","<<y<<"\n";

		Body b(x,y, temp); // Starts all bodies in the center of the screen
		sf::CircleShape shape(8, 3);

		// Changing the Visual Properties of the shape.
		shape.setPosition(b.location.x, b.location.y); // Sets position of shape to random location that body was set to.
		shape.setOrigin(12, 12);
		//shape.setPosition(window_width, window_height); // Testing purposes, starts all shapes in the center of screen.
		shape.setFillColor(sf::Color::Yellow);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1);
		shape.setRadius(bodiesSize);

		// Adding the body to the flock and adding the shapes to the vector<sf::CircleShape>
		flock.addBody(b);
		shapes.push_back(shape);
		window.draw(shape);

		if(y == 500) //increments the  x pos so bots are drawn in a grid.
		{
			x+=50;
		}

	}
	window.display();
	sleep(1);
	ros::Publisher pub = _n.advertise < wvu_swarm_std_msgs::vicon_bot_array > ("vicon_array", 1000); //Publishes like Vicon
	ros::Subscriber sub = _n.subscribe("final_execute", 1000, &Sim::vectorCallback, this); //subscribes to funnel
	ros::Rate loopRate(10);
	//publishes initial information for each bot
	wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();
	pub.publish(vb_array);
	ros::spinOnce();
	while (window.isOpen() && ros::ok())
	{
		HandleInput();
		Render();
		wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();

		//publishing vicon_bot_array
		//flock.printMessage(vb_array);
		pub.publish(vb_array);
		ros::spinOnce();
		loopRate.sleep();
	}
}

void Sim::HandleInput()
{
	sf::Event event;
	while (window.pollEvent(event))
	{
		// Pressing the escape key will close the program
		if ((event.type == sf::Event::Closed)
				|| (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
		{
			window.close();
		}

		bool pauseSim = false;
		if((event.type == sf::Event::KeyPressed )&&( event.key.code == sf::Keyboard::Pause)){
			pauseSim = true;
			std::cout<<"paused"<<std::endl;
		}
		while(pauseSim == true) //runs while pause in effect.
		{
			if(window.pollEvent(event))
			{
				if ((event.type == sf::Event::KeyPressed )&&( event.key.code == sf::Keyboard::Pause))
				{
					pauseSim = false;
					std::cout<<"unpaused"<<std::endl;
				}
				if ((event.type == sf::Event::Closed) || (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
				{
					window.close();
				}
		  }
		}
	}

}

void Sim::Render()
{
	window.clear();
	flock.flocking();
// Draws all of the bodies out, and applies functions that are needed to update.
	for (int i = 0; i < shapes.size(); i++)
	{

		// Matches up the location of the shape to the body
		shapes[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);

		// Calculates the angle where the velocity is pointing so that the triangle turns towards it.
		float theta = 180 / M_PI * flock.getBody(i).angle(flock.getBody(i).velocity);
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
		flock.flock.at(i).updatedCommand = false;
		flock.flock.at(i).updatedPosition = false;
	}



	window.display(); //updates display

}

