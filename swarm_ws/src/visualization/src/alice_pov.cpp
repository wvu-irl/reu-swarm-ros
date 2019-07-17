#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <math.h>
#include <wvu_swarm_std_msgs/alice_mail_array.h>
#include <swarm_simulation/Font.h>
#include <visualization/alice_pov.h>
#include <alice_swarm/get_map.h>
bool overlap = true;

//#include "ros/ros.h"
void AlicePOV::mailCallback(const wvu_swarm_std_msgs::alice_mail_array &msg)
{
	mail = msg;
}

// Construct window using SFML
AlicePOV::AlicePOV(void)
{
	bodiesSize = 10.5;
	sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	window_height = 600;
	window_width = 600;
	this->window.create(sf::VideoMode(window_width, window_height, desktop.bitsPerPixel), "Alice POV",
			sf::Style::Titlebar);
	this->window2.create(sf::VideoMode(300, window_height, desktop.bitsPerPixel), "Alice POV", sf::Style::Titlebar);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void AlicePOV::Run(ros::NodeHandle _n)
{
	//Initializes all publishers and subscribers

	ros::Subscriber sub = _n.subscribe("alice_mail_array", 1000, &AlicePOV::mailCallback, this);
	ros::ServiceClient client = _n.serviceClient < alice_swarm::get_map > ("get_map");
	ros::Rate loopRate(15);

	while (window.isOpen() && ros::ok()) //main while loop (runs the simulation).
	{
		HandleInput();
		Render(client);
		ros::spinOnce();
		loopRate.sleep();
	}
}
void AlicePOV::HandleInput() //switches the robot viewed depending on what buttons are pressed
{
	sf::Event event;
	while (window.pollEvent(event) || window2.pollEvent(event))
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
void AlicePOV::drawMail()
{
	for (int i = 0; i < mail.mails.size(); i++)
	{
		if (mail.mails.at(i).name == name)
		{
			for (int j = 0; j < mail.mails.at(i).obsMail.size(); j++)
			{
				wvu_swarm_std_msgs::ellipse temp = mail.mails.at(i).obsMail.at(j);
				unsigned short quality = 70;
				sf::ConvexShape ellipse;
				ellipse.setPointCount(quality);
				for (unsigned short i = 0; i < quality; ++i)
				{
					float rad = (360 / quality * i) / (360 / M_PI / 2);
					float x = 3 * cos(rad) * temp.x_rad;
					float y = 3 * sin(rad) * temp.y_rad;
					float newx = x * cos(-temp.theta_offset) - y * sin(-temp.theta_offset);
					float newy = x * sin(-temp.theta_offset) + y * cos(-temp.theta_offset);
					ellipse.setPoint(i, sf::Vector2f(newx, newy));
				}

				ellipse.setPosition(300 + 3 * temp.offset_x, 300 - 3 * temp.offset_y);
				ellipse.setFillColor(sf::Color::Yellow);
				ellipse.setOutlineColor(sf::Color::White);
				ellipse.setOutlineThickness(1);
				window.draw(ellipse);
			}

			for (int j = 0; j < mail.mails.at(i).neighborMail.size(); j++)
			{
				wvu_swarm_std_msgs::neighbor_mail temp = mail.mails.at(i).neighborMail.at(j);
				sf::CircleShape shape(0);
				// Changing the Visual Properties of the (neighboring) robot
				shape.setPosition(300 + 3 * temp.x, 300 - 3 * temp.y);
				shape.setOrigin(bodiesSize, bodiesSize);
				float inten = 10 * mail.mails.at(temp.name).contVal;
				if (inten > 255)
					inten = 255;
				shape.setFillColor(sf::Color(255 - (int) inten, 0, (int) inten, 255));
				shape.setOutlineColor(sf::Color::White);
				shape.setOutlineThickness(1);
				shape.setRadius(bodiesSize);
				window.draw(shape);

				sf::RectangleShape line(sf::Vector2f(5, 2));
				line.setFillColor(sf::Color::Black);
				line.setPosition(300 + 3 * temp.x, 300 - 3 * temp.y);
				line.setOrigin(-2, 1);
				line.setRotation(180.0 / M_PI * (-temp.ang));
				window.draw(line);
			}
			for (int j = 0; j < mail.mails.at(i).targetMail.size(); j++)
			{
				wvu_swarm_std_msgs::point_mail temp = mail.mails.at(i).targetMail.at(j);
				sf::CircleShape shape(0);
				// Changing the Visual Properties of the obstacle
				shape.setPosition(300 + 3 * temp.x, 300 - 3 * temp.y);
				shape.setOrigin(bodiesSize, bodiesSize);
				shape.setFillColor(sf::Color::Green);
				shape.setRadius(bodiesSize);
				shape.setOutlineColor(sf::Color::White);
				shape.setOutlineThickness(1);
				window.draw(shape);
			}
			for (int j = 0; j < mail.mails.at(i).flowMail.size(); j++)
			{
				wvu_swarm_std_msgs::flow_mail temp = mail.mails.at(i).flowMail.at(j);

				sf::RectangleShape line(sf::Vector2f(temp.pri * 10, 1));
				line.setFillColor(sf::Color::Cyan);
				line.setPosition(300 + 3 * temp.x, 300 - 3 * temp.y);
				line.setRotation(180.0 / M_PI * (-temp.dir));
				window.draw(line);
				line.setOutlineColor(sf::Color::White);
				line.setOutlineThickness(1);
			}
			sf::CircleShape shape(0);
			// Changing the Visual Properties of the robot
			shape.setPosition(300, 300); // Sets position of shape to the middle
			shape.setOrigin(bodiesSize, bodiesSize);
			float inten = 10 * mail.mails.at(name).contVal;
			if (inten > 255)
				inten = 255;
			shape.setFillColor(sf::Color(255 - (int) inten, 0, (int) inten, 255));
			shape.setOutlineColor(sf::Color::White);
			shape.setOutlineThickness(1);
			shape.setRadius(bodiesSize);
			window.draw(shape);

			sf::RectangleShape line(sf::Vector2f(5, 2));
			line.setFillColor(sf::Color::Black);
			line.setPosition(300, 300);
			line.setOrigin(-2, 1);
			line.setRotation(0);
			window.draw(line);
		}
	}
}
void AlicePOV::drawMsg(ros::ServiceClient _client)
{
	sf::Color gray(100, 100, 100); //the border color for objects found on the map
	alice_swarm::get_map srv;
	srv.request.name = name;
	_client.call(srv);
	wvu_swarm_std_msgs::map map = srv.response.map;

	for (int j = 0; j < map.obsMsg.size(); j++)
	{
		wvu_swarm_std_msgs::ellipse temp = map.obsMsg.at(j).ellipse;
		unsigned short quality = 70;
		sf::ConvexShape ellipse;
		ellipse.setPointCount(quality);
		for (unsigned short i = 0; i < quality; ++i)
		{
			float rad = (360 / quality * i) / (360 / M_PI / 2);
			float x = 3 * cos(rad) * temp.x_rad;
			float y = 3 * sin(rad) * temp.y_rad;
			float newx = x; //* cos(map.oheading-temp.theta_offset) - y * sin(map.oheading-temp.theta_offset);
			float newy = y;// * sin(map.oheading-temp.theta_offset) + y * cos(map.oheading-temp.theta_offset);
			ellipse.setPoint(i, sf::Vector2f(newx, newy));
		}

		ellipse.setPosition(150 + 3 * (map.ox + cos(map.oheading) * temp.offset_x - sin(map.oheading) * temp.offset_y),
				300 - 3 * (map.oy + sin(map.oheading) * temp.offset_x + cos(map.oheading) * temp.offset_y));
		ellipse.setFillColor(sf::Color::Yellow);
		ellipse.setOutlineColor(gray);
		ellipse.setOutlineThickness(1);
		window2.draw(ellipse);
	}

	for (int j = 0; j < map.tarMsg.size(); j++)
	{
		wvu_swarm_std_msgs::point_mail temp = map.tarMsg.at(j).pointMail;
		sf::CircleShape shape(0);
		// Changing the Visual Properties of the target
		shape.setPosition(150 + 3 * (map.ox + cos(map.oheading) * temp.x - sin(map.oheading) * temp.y),
				300 - 3 * (map.oy + sin(map.oheading) * temp.x + cos(map.oheading) * temp.y));
		shape.setOrigin(bodiesSize, bodiesSize);
		shape.setFillColor(sf::Color::Green);
		shape.setRadius(bodiesSize);
		shape.setOutlineColor(gray);
		shape.setOutlineThickness(1);
		window2.draw(shape);
	}
	for (int j = 0; j < map.contMsg.size(); j++)
	{
		wvu_swarm_std_msgs::point_mail temp = map.contMsg.at(j).pointMail;
		sf::CircleShape shape(0);
		// Changing the Visual Properties of the contour point
		shape.setPosition(150 + 3 * (map.ox + cos(map.oheading) * temp.x - sin(map.oheading) * temp.y),
				300 - 3 * (map.oy + sin(map.oheading) * temp.x + cos(map.oheading) * temp.y));
		shape.setOrigin(2, 2);
		shape.setRadius(2);
		float inten = 10 * map.contMsg.at(j).contVal;
		if (inten > 255)
			inten = 255;
		shape.setFillColor(sf::Color(255 - (int) inten, 0, (int) inten, 255));
		//no outline cause the point is tiny...
//		shape.setOutlineColor(gray);
//		shape.setOutlineThickness(1);
		window2.draw(shape);
	}

}

void AlicePOV::Render(ros::ServiceClient _client) //draws changes in simulation states to the window.
{

	window.clear();
	window2.clear();
	drawMail();
	drawMsg(_client);
	window.display(); //updates display
	window2.display();
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

