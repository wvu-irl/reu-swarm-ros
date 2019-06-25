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
bool update;
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

					flock.flock.at(i).heading = fmod(flock.flock.at(i).heading + 2 * M_PI, 2 * M_PI);

					float a;
					float b;
					float temp_r;

					float theta = msg.commands.at(j).theta / 180 * M_PI;
					float r = msg.commands.at(j).r;
					if (0 < theta && theta < M_PI)
					{
						a = 1;
						b = r * cos(theta / 2);
						temp_r = b;
						flock.flock.at(i).heading += M_PI / 90 * theta;
					} else if (M_PI < theta && theta < 2 * M_PI)
					{
						b = 1;
						a = -r * cos(theta / 2);
						temp_r = a;
						flock.flock.at(i).heading -= M_PI / 90 * (2 * M_PI - theta);
					}

					float x = temp_r * cos(flock.flock.at(i).heading);
					float y = temp_r * sin(flock.flock.at(i).heading);

//					float theta = msg.commands.at(j).theta / 180 * M_PI;
//					if (theta > M_PI / 24 && theta < M_PI)
//					{
//						flock.flock.at(i).velocity.set(0, 0);
//						flock.flock.at(i).heading += M_PI / 24;
//
//					} else if (theta < 47 * M_PI / 24 && theta > M_PI)
//					{
//						flock.flock.at(i).velocity.set(0, 0);
//						flock.flock.at(i).heading -= M_PI / 24;
//					} else
//					{
//						flock.flock.at(i).heading += theta;
//						flock.flock.at(i).velocity.set(1 * msg.commands.at(j).r * cos(flock.flock.at(i).heading),
//								-1 * msg.commands.at(i).r * sin(flock.flock.at(i).heading));
//					}

//					if(flock.flock.at(i).collision == false)
//					{
					flock.flock.at(i).heading = fmod(flock.flock.at(i).heading, (2 * M_PI));
					flock.flock.at(i).velocity.set(x, -y);

					if (msg.commands.at(j).r < 0.01)
					{
						flock.flock.at(i).velocity.set(0, 0);
					}
					flock.flock.at(i).updatedCommand = true;
				}
			}
		}
	}
}

void Sim::obsCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
	obstacles.clear();
	for (int i = 0; i < msg.point.size(); i++)
	{
		obstacles.push_back(msg.point.at(i));
	}
}
void Sim::targetCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
	if (!update)
	{
		targets = msg;
		update = true;
	}
}
void Sim::flowCallback(const wvu_swarm_std_msgs::flows &msg)
{
	flows = msg;
}

// Construct window using SFML
Sim::Sim()
{
	this->bodiesSize = 7.5; //7.5
	sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	this->window_height = 600;
	this->window_width = 300;
	game = false;
	//std::cout<<"Window h;w = "<<window_height << "; "<<window_width<<"\n";
	this->window.create(sf::VideoMode(window_width, window_height, desktop.bitsPerPixel), "Swarm Simulation",
			sf::Style::None);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void Sim::Run(ros::NodeHandle _n)
{
	PrevIteration pI
	{ false, 0, false, false, false }; //struct for storing click and drag info.
	std::cout << "initialized pI\n";
	char letters[100] =
	{ 'D', 'E', 'P', 'A', 'N', 'J', 'G', 'A', 'C', 'T', 'M', 'A', 'M', 'D', 'S', 'C', 'N', 'H', 'V', 'A', 'N', 'Y', 'N',
			'C', 'R', 'I', 'V', 'T', 'K', 'Y', 'T', 'N', 'O', 'H', 'L', 'A', 'I', 'N', 'M', 'S', 'I', 'L', 'A', 'L', 'M', 'E',
			'M', 'O', 'A', 'R', 'M', 'I', 'F', 'L', 'T', 'X', 'I', 'A', 'W', 'I', 'C', 'A', 'M', 'N', 'O', 'R', 'K', 'A', 'W',
			'V', 'N', 'V', 'N', 'E', 'C', 'O', 'N', 'D', 'S', 'D', 'M', 'T', 'W', 'A', 'I', 'D', 'W', 'Y', 'U', 'T', 'O', 'K',
			'N', 'M', 'A', 'Z', 'A', 'K', 'H', 'I' };

	int x = 50; //x inital positions for the bots.

	for (int i = 0; i < 10; i++)
	{
		char temp[2] =
		{ letters[2 * i], letters[2 * i + 1] };
		int y = 50 * (i % 10) + 50; // y pos of bots

//		std::cout<<"----Bot ID: "<<temp[0]<<temp[1]<<"-------\n";
//		std::cout<<"x,y: "<<x<<","<<y<<"\n";

		Body b(x, y, temp); // Starts all bodies in the center of the screen
		b.sid = i % 2 + 1;
		sf::CircleShape shape(0);

		// Changing the Visual Properties of the shape.
		shape.setPosition(b.location.x, b.location.y); // Sets position of shape to random location that body was set to.
		shape.setOrigin(7.5, 7.5);
		//shape.setPosition(window_width, window_height); // Testing purposes, starts all shapes in the center of screen.
		if (b.sid == 1)
			shape.setFillColor(sf::Color::Yellow);
		else
			shape.setFillColor(sf::Color::Cyan);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1);
		shape.setRadius(bodiesSize);
		shape.rotate((180.0 / M_PI) * M_PI_2);

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

		if (y == 500) //increments the  x pos so bots are drawn in a grid.
		{
			x += 50;
		}

	}
	window.display();

	//Initializes all publishers and subscribers
	ros::Publisher pub = _n.advertise < wvu_swarm_std_msgs::vicon_bot_array > ("vicon_array", 1000); //Publishes like Vicon
	ros::Publisher pub2 = _n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_targets", 1000);
	ros::Subscriber sub = _n.subscribe("final_execute", 1000, &Sim::vectorCallback, this); //subscribes to funnel
	ros::Subscriber sub2 = _n.subscribe("virtual_obstacles", 1000, &Sim::obsCallback, this); //subscribes to virtual obstacles
	ros::Subscriber sub3 = _n.subscribe("virtual_targets", 1000, &Sim::targetCallback, this); //gets virtual targets
	ros::Subscriber sub4 = _n.subscribe("virtual_flows", 1000, &Sim::flowCallback, this); //gets virtual targets
	ros::Rate loopRate(50);

	//publishes initial information for each bot
	wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();
	pub.publish(vb_array);
	ros::spinOnce();

	while (window.isOpen() && ros::ok()) //main while loop (runs the simulation).
	{
		while(game == false && window.isOpen() && ros::ok())//secondary loop allows for winning condition.
		{
			pI = HandleInput(pI);
			Render();
			wvu_swarm_std_msgs::vicon_bot_array vb_array = flock.createMessages();
			//publishing vicon_bot_array
			pub.publish(vb_array);
		//		if (targets.point.size() > 0)
		//		{
		//			targets.point.at(0).x = vb_array.poseVect.at(0).botPose.transform.translation.x;
		//			targets.point.at(0).y = vb_array.poseVect.at(0).botPose.transform.translation.y;
		//			//targets.point.at(0).sid=0;
		//		}

			if (!update) pub2.publish(targets);

			ros::spinOnce();
			std::cout<<"iteration complete"<<std::endl;
			loopRate.sleep();
		}
		//---------=Code for winning the game=---------------
		if (game == true)
		{
			for (int j = 0; j<2; j++)
			{
				std::cout<<winner<<" has won the game!!!!!!!!!!!!"<<std::endl;
			}
			game = false;
			for(int i = 0; i < targets.point.size(); i ++)
			{
				targets.point.at(0).x = 0;
				targets.point.at(0).y = 0;
				pI.dragging = false;
				pI.prevClick = false;
			}
		}
		//-------------------=End=---------------
	}
}

PrevIteration Sim::HandleInput(PrevIteration _pI)		//handels input to the graphics window
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
		if ((event.type == sf::Event::Closed)
				|| (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
		{
			window.close();
		} //-----------------------------------------------------------

		//------------------allows for pause. Press the Pause button.-----------
		pauseSim = pause(event.type == sf::Event::KeyPressed, event.key.code == sf::Keyboard::Space, pauseSim, &window,
				event);


		//----------Allows for click and drag for bots. ------------------------------
		clickNdragBots(&_pI, mX, mY, event);
		//----------------------- End --------------------------------------

		// ---------------------- Click and Drag for the target ----------------------------------------
		clickNdragTarget(&_pI, mX, mY, event);
		//----------------------------- End-------------------------------------
	}
	return _pI; //tracks state of dragging (see sim.h)

//	// Checks or A to be pressed, draws and adds bodies to flock if so.
//	  if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
//	        // Gets mouse coordinates, sets that as the location of the body and the shape
//	        sf::Vector2i mouseCoords = sf::Mouse::getPosition(window);
//	        Body b(mouseCoords.x, mouseCoords.y, false);
//	        sf::CircleShape shape(4);
//
//	        // Changing visual properties of newly created body
//	        shape.setPosition(mouseCoords.x, mouseCoords.y);
//	        shape.setOutlineColor(sf::Color::White);
//	        shape.setFillColor(sf::Color::White);
//	        shape.setOutlineColor(sf::Color::White);
//	        shape.setOutlineThickness(1);
//	        shape.setRadius(bodiesSize);
//
//	        // Adds newly created body and shape to their respective data structure
//	        flock.addBody(b);
//	        shapes.push_back(shape);
//
//	        // New Shape is drawn
//	        window.draw(shapes[shapes.size() - 1]);
//	    }
//
//	    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Delete)) {
//	          // Gets mouse coordinates, sets that as the location of the body and the shape
//	          sf::Vector2i mouseCoords = sf::Mouse::getPosition(window);
//	          Body b(mouseCoords.x, mouseCoords.y, false);
//	          sf::CircleShape shape(4);
//
//	          // Changing visual properties of newly created body
//	          shape.setPosition(mouseCoords.x, mouseCoords.y);
//	          shape.setOutlineColor(sf::Color::White);
//	          shape.setFillColor(sf::Color::White);
//	          shape.setOutlineColor(sf::Color::White);
//	          shape.setOutlineThickness(1);
//	          shape.setRadius(bodiesSize);
//
//	          //deletes bodies from the second through the hird element
//	          shapes.erase(shapes.begin() + 1, shapes.begin() + 2);
//


}

void Sim::clickNdragBots(PrevIteration *_pI, float _mX, float _mY, sf::Event _event)//for click and drag on bots
{
	int i = 0; //iterator for dragging while loop
	bool found = false; //sentinel for finding a selected bot.

	//----------Allows for click and drag for bots. ------------------------------
	if (_pI->dragging == true && _pI->bot == true)
	{
		flock.flock.at(_pI->botId).location.x = sf::Mouse::getPosition(window).x; //event.mouseButton.x;
		flock.flock.at(_pI->botId).location.y = sf::Mouse::getPosition(window).y; //event.mouseButton.y;
	}
	if (_event.type == sf::Event::MouseButtonPressed && _event.mouseButton.button == sf::Mouse::Left
			&& _pI->prevClick == true && _pI->bot == true)
	{
		_pI->dragging = false;
		_pI->prevClick = false;
		_pI->bot = false;
	} else if (_event.type == sf::Event::MouseButtonPressed && _event.mouseButton.button == sf::Mouse::Left
			&& _pI->prevClick == false && _pI->bot == false && _pI->target != true)
	{
		while (found != true)
		{
			if (((flock.flock.at(i).location.x > _mX - 6) && (flock.flock.at(i).location.x < _mX + 6))
					&& ((flock.flock.at(i).location.y > _mY - 6) && (flock.flock.at(i).location.y < _mY + 6)))
			{
				found = true;
				_pI->botId = i;
				_pI->dragging = true;
				_pI->prevClick = true;
				_pI->bot = true;
			} else if (i == flock.flock.size() - 1)
			{
				found = true;
			}
			i++;
		}
	}
}

void Sim::clickNdragTarget(PrevIteration *_pI, float _mX, float _mY, sf::Event _event)//for click and drag on targets
{
	int i = 0; //iterator for dragging while loop
	bool found = false; //sentinel for finding a selected bot.
	//		----------Allows for click and drag. ------------------------------
	if (_pI->dragging == true && _pI->target==true)
	{
		targets.point.at(i).x = sf::Mouse::getPosition(window).x / 3 - 50;
		; //event.mouseButton.x;
		targets.point.at(i).y = sf::Mouse::getPosition(window).y / -3 + 100; //event.mouseButton.y;
		std::cout << "yo" << std::endl;
	}
	if (_event.type == sf::Event::MouseButtonPressed && _event.mouseButton.button == sf::Mouse::Left
			&& _pI->prevClick == true && _pI->target == true)
	{
		_pI->dragging = false;
		_pI->prevClick = false;
		_pI->target = false;

	} else if (_event.type == sf::Event::MouseButtonPressed && _event.mouseButton.button == sf::Mouse::Left
			&& _pI->prevClick == false && _pI->target == false && _pI->bot !=true)
	{
		while (found != true && i < targets.point.size())
		{
			if (((targets.point.at(i).x > _mX / 3 - 50 - 6) && (targets.point.at(i).x < _mX / 3 - 50 + 6))
					&& ((targets.point.at(i).y > _mY / -3 + 100 - 6) && (targets.point.at(i).y < _mY / -3 + 100 + 6)))
			{
				found = true;
				_pI->botId = i;
				_pI->dragging = true;
				_pI->prevClick = true;
				_pI->target = true;
			}
			i++;
		}
	}
}

void Sim::Render() //draws changes in simulation states to the window.
{
	window.clear();
	drawGoals();
	flock.flocking(&targets);
	if (update)
	{
		updateTargetPos();
		update = false;
	}
	drawObstacles();
	drawTargets();
	//drawFlows();

// Draws all of the bodies out, and applies functions that are needed to update.
	for (int i = 0; i < shapes.size(); i++)
	{ // Matches up the location of the shape to the body
		shapes[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);
		lines[i].setPosition(flock.getBody(i).location.x, flock.getBody(i).location.y);

		float theta = 180.0 / M_PI * (flock.flock.at(i).heading);
		shapes[i].setRotation(90 - theta); //alignes body with direction of motion
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
{ //checks if pause pressed. Inf loop if so.
	if ((_key_pressed) && (_pause_pressed))
	{
		_pause_sim = true;
		std::cout << "paused" << std::endl;
	}
	while (_pause_sim == true) //runs while pause in effect.
	{
		if (win->pollEvent(_event))
		{
			if ((_event.type == sf::Event::KeyPressed) && (_event.key.code == sf::Keyboard::Space))
			{ //allows for unpause.
				_pause_sim = false;
				std::cout << "unpaused" << std::endl;
			}
			if ((_event.type == sf::Event::Closed)
					|| (_event.type == sf::Event::KeyPressed && _event.key.code == sf::Keyboard::Escape))
			{
				win->close();
			}
		}
	}
	return _pause_sim;
}

void Sim::drawFlows()//draws flows (basically a vector field)
{
	float x2;
	float y2;

	//std::cout<<"number of flows: "<<flows.flow.size()<<std::endl;
	for (int i = 0; i < flows.flow.size(); i++)
	{
		sf::RectangleShape line(sf::Vector2f(flows.flow.at(i).r, 1));
		line.setPosition(flows.flow.at(i).x * 3 + 150, 300 - flows.flow.at(i).y * 3);

		x2 = flows.flow.at(i).r * cos(flows.flow.at(i).theta);
		y2 = flows.flow.at(i).r * sin(flows.flow.at(i).theta);

		if (y2 < 0)
		{
			y2 = -y2;
		} else
		{
			y2 = -y2;
		}

		//convert to sim frame
		line.setFillColor(sf::Color::White);
		line.setOrigin(0, 0);
		line.setRotation(-flows.flow.at(i).theta * 180 / M_PI);
		line.setOutlineColor(sf::Color::Black);
		line.setOutlineThickness(1);
		window.draw(line);

		sf::RectangleShape line1(sf::Vector2f(1, 1)); //gives vector direction
		line1.setFillColor(sf::Color::Red);
		line1.setPosition(flows.flow.at(i).x * 3 + 150 + x2, 300 + y2 - flows.flow.at(i).y * 3);
		line1.setOrigin(-1, 0);
		line1.setRotation(-flows.flow.at(i).theta * 180 / M_PI + 130);
		line1.setOutlineColor(sf::Color::Black);
		line1.setOutlineThickness(1);
		window.draw(line1);

//		sf::RectangleShape line2(sf::Vector2f(7, 1));
//		line2.setFillColor(sf::Color::White);
//		line2.setPosition(flows.flow.at(i).x*3 + 150 + x2, 300 + y2 - flows.flow.at(i).y*3);
//		line2.setOrigin(-1, 1);
//		line2.setRotation(-flows.flow.at(i).theta*180/M_PI - 130);
//		line2.setOutlineColor(sf::Color::Black);
//		line2.setOutlineThickness(1);
//		window.draw(line2);
	}
}

void Sim::drawTargets() //draws targets
{
	for (int i = 0; i < targets.point.size(); i++) //draws targets
	{
		sf::CircleShape shape(0);
		shape.setPosition(targets.point.at(i).x * 3 + 150, 300 - targets.point.at(i).y * 3); // Sets position of shape to random location that body was set to.
		shape.setOrigin(10, 10);
		shape.setFillColor(sf::Color::Green);
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(1);
		shape.setRadius(10);
		window.draw(shape);
	}
}

void Sim::drawObstacles()//draws obstacles
{
	for (int i = 0; i < obstacles.size(); i++)
	{
		sf::CircleShape shape(0);
		shape.setPosition(obstacles.at(i).x * 3 + 150, 300 - obstacles.at(i).y * 3); // Sets position of shape to random location that body was set to.
		shape.setOrigin(2, 2);
		shape.setFillColor(sf::Color::Blue);
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(1);
		shape.setRadius(2);
		window.draw(shape);
	}
}

//-#######################Air Hockey Specific Functions ###################################

void Sim::updateTargetPos() //specificly a free particle (puck) with damping. For Airhockey.
{
	for (int i = 0; i < targets.point.size(); i++)
	{
		winCheck(i);

		if (targets.point.at(i).x < -45 || targets.point.at(i).x > 45)
			targets.point.at(i).vx *= -1;
		if (targets.point.at(i).y < -95 || targets.point.at(i).y > 95)
			targets.point.at(i).vy *= -1;

//		float heyX = 0;
//		float heyY = 0;
//
//		if (targets.point.at(i).vx > 0){heyX = 0.1;}
//		else if (targets.point.at(i).vx < 0){heyX = -0.1;}
//
//		if (targets.point.at(i).vy > 0){heyX = 0.1;}
//		else if (targets.point.at(i).vy < 0){heyX = -0.1;}
//
//		targets.point.at(i).x += heyX;
//		targets.point.at(i).y += heyY;

		targets.point.at(i).x += targets.point.at(i).vx;
		targets.point.at(i).y += -targets.point.at(i).vy;

		targets.point.at(i).vx *= 0.99;
		targets.point.at(i).vy *= 0.99;
	}
}

void Sim::winCheck(int i) //checks if the game has been won.
{
	if(targets.point.at(i).x > -15 && targets.point.at(i).x < 15)
	{
		if(targets.point.at(i).y < -94)
		{
			winner = "The Blue Team";
			game = true;
		}
		else if(targets.point.at(i).y > 94)
		{
			winner = "The Yellow Team";
			game = true;
		}
		else
		{
			game = false;
		}
	}
}

void Sim::drawGoals() //draws the goal posts for airhockey.
{
	sf::RectangleShape g1(sf::Vector2f(5, 5));
	sf::RectangleShape g2(sf::Vector2f(5, 5));
	sf::RectangleShape g3(sf::Vector2f(5, 5));
	sf::RectangleShape g4(sf::Vector2f(5, 5));

	g1.setPosition(100,3);
	g2.setPosition(200,3);
	g3.setPosition(100,595);
	g4.setPosition(200,595);

	g1.setOutlineColor(sf::Color::White);
	g2.setOutlineColor(sf::Color::White);
	g3.setOutlineColor(sf::Color::White);
	g4.setOutlineColor(sf::Color::White);

	window.draw(g1);
	window.draw(g2);
	window.draw(g3);
	window.draw(g4);
}
