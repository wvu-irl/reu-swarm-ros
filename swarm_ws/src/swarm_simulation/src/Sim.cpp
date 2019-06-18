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
							-1 * msg.commands.at(i).r
									* sin(flock.flock.at(i).angle(flock.flock.at(i).velocity) + msg.commands.at(j).theta));
					//		}
					flock.flock.at(i).heading += msg.commands.at(j).theta;
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
	this->bodiesSize = 7.5;
	sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	this->window_height = 600;
	this->window_width = 300;

	//std::cout<<"Window h;w = "<<window_height << "; "<<window_width<<"\n";
	this->window.create(sf::VideoMode(window_width, window_height, desktop.bitsPerPixel),
			"Swarm Simulation", sf::Style::None);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void Sim::Run(ros::NodeHandle _n)
{
	PrevIteration pI{false,0,false};//struct for storing click and drag info.
	std::cout<<"initialized pI\n";
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
		sf::CircleShape shape(0);

		// Changing the Visual Properties of the shape.
		shape.setPosition(b.location.x, b.location.y); // Sets position of shape to random location that body was set to.
		shape.setOrigin(7.5, 7.5);
		//shape.setPosition(window_width, window_height); // Testing purposes, starts all shapes in the center of screen.
		shape.setFillColor(sf::Color::Yellow);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1);
		shape.setRadius(bodiesSize);
		shape.rotate((180.0/M_PI)*M_PI_2);

		//creates the red line at the head of each bot.
		sf::RectangleShape line(sf::Vector2f(5, 2));
		line.setFillColor(sf::Color::Red);
		line.setPosition(b.location.x, b.location.y);
		line.setOrigin(-2, 1);

		// Adding the body to the flock and adding the shapes to the vector<sf::CircleShape>
		flock.addBody(b);
		shapes.push_back(shape);

		//saves a vector of lines (one for each bot).
		lines.push_back(line);

		//draw all obejcts on window.
		window.draw(shape);
    window.draw(line);

		if(y == 500) //increments the  x pos so bots are drawn in a grid.
		{
			x+=50;
		}

	}
	window.display();
	sleep(1);
	ros::Publisher pub = _n.advertise < wvu_swarm_std_msgs::vicon_bot_array > ("vicon_array", 1000); //Publishes like Vicon
	ros::Subscriber sub = _n.subscribe("final_execute", 1000, &Sim::vectorCallback, this); //subscribes to funnel
	ros::Rate loopRate(50);

	//publishes initial information for each bot
	wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();
	pub.publish(vb_array);
	ros::spinOnce();
	while (window.isOpen() && ros::ok())
	{
		pI = HandleInput(pI);
		Render();
		wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();

		//publishing vicon_bot_array
		//flock.printMessage(vb_array);
		pub.publish(vb_array);
		ros::spinOnce();
		loopRate.sleep();
	}
}

PrevIteration Sim::HandleInput(PrevIteration _pI)//handels input to the graphics window
{
	sf::Event event;
	while (window.pollEvent(event))
	{
		int i = 0; //iterator for dragging while loop
		bool found = false; //sentinel for finding a selected bot.
		float mX = event.mouseButton.x; //mouse x pos
		float mY = event.mouseButton.y; //mouse y pos
		bool pauseSim = false; //pause boolean.

		//---------- Pressing the escape key will close the program
		if ((event.type == sf::Event::Closed)|| (event.type == sf::Event::KeyPressed
				&& event.key.code == sf::Keyboard::Escape))
		{
			window.close();
		}//-----------------------------------------------------------

		//------------------allows for pause. Press the Pause button.-----------
		pauseSim = pause(event.type == sf::Event::KeyPressed, event.key.code == sf::Keyboard::Space,
				pauseSim, &window, event);

		//----------Allows for click and drag. ------------------------------
		if (_pI.dragging == true)
		{
			flock.flock.at(_pI.botId).location.x = sf::Mouse::getPosition(window).x;//event.mouseButton.x;
			flock.flock.at(_pI.botId).location.y = sf::Mouse::getPosition(window).y;//event.mouseButton.y;
		}
		if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left
				&& _pI.prevClick == true)
		{
			_pI.dragging = false;
			_pI.prevClick = false;
		}
		else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left
				&& _pI.prevClick == false)
		{
			 while(found != true)
			 {
				 if (((flock.flock.at(i).location.x > mX - 6 ) && (flock.flock.at(i).location.x < mX + 6))
						 && ((flock.flock.at(i).location.y > mY - 6) && (flock.flock.at(i).location.y < mY + 6)))
				 {
					 found = true;
					 _pI.botId = i;
					 _pI.dragging = true;
					 _pI.prevClick = true;
				 }
				 else if(i==49)
				 {
					 found = true;
				 }
				 i++;
			 }
		}//-----------------------------------------------------------------------------------------
	}
	return _pI; //tracks state of dragging (see sim.h)
}

void Sim::Render() //draws changes in simulation states to the window.
{
	window.clear();
	flock.flocking();
// Draws all of the bodies out, and applies functions that are needed to update.
	for (int i = 0; i < shapes.size(); i++)
	{// Matches up the location of the shape to the body
		shapes[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);
		lines[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);

		float theta = 180.0 / M_PI * (flock.flock.at(i).heading);
		shapes[i].setRotation(90-theta); //alignes body with direction of motion
		lines[i].setRotation(-theta); //alignes line with direction of motion
		//^for some reason, sfml has clockwise as +theta direction.

		window.draw(shapes[i]);
		window.draw(lines[i]);
		flock.flock.at(i).updatedCommand = false;
		flock.flock.at(i).updatedPosition = false;
	}
	window.display(); //updates display
}

bool Sim::pause(bool _key_pressed, bool _pause_pressed, bool _pause_sim, sf::RenderWindow* win, sf::Event _event)
{//checks if pause pressed. Inf loop if so.
	if(( _key_pressed)&&(_pause_pressed))
	{
		_pause_sim = true;
		std::cout<<"paused"<<std::endl;
	}
	while(_pause_sim == true) //runs while pause in effect.
	{
		if(win->pollEvent(_event))
		{
			if ((_event.type == sf::Event::KeyPressed )&&( _event.key.code == sf::Keyboard::Space))
			{//allows for unpause.
				_pause_sim = false;
				std::cout<<"unpaused"<<std::endl;
			}
			if ((_event.type == sf::Event::Closed) || (_event.type == sf::Event::KeyPressed && _event.key.code == sf::Keyboard::Escape))
			{
				win->close();
			}
		}
	}
	return _pause_sim;
}


