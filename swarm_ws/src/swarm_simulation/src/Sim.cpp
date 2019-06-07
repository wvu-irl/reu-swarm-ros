#include <iostream>
#include <swarm_simulation/Flock.h>
#include <swarm_simulation/Body.h>
#include <swarm_simulation/Pvector.h>
#include <swarm_simulation/Sim.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
//#include "ros/ros.h"

// Construct window using SFML
Sim::Sim()
{
    this->bodiesSize = 11.0;
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    this->window_height = desktop.height;
    this->window_width  = desktop.width;
    this->window.create(sf::VideoMode(window_width, window_height, desktop.bitsPerPixel), "Flocking Simulation Aleks Hatfield", sf::Style::None);
}

// Run the simulation. Run creates the bodies that we'll display, checks for user
// input, and updates the view
void Sim::Run()
{
    for (int i = 0; i < 49; i++) {
        Body b(window_width / 2 , window_height / 2 ); // Starts all bodies in the center of the screen
        sf::CircleShape shape(4);

        // Changing the Visual Properties of the shape
        // shape.setPosition(b.location.x, b.location.y); // Sets position of shape to random location that body was set to.
        shape.setPosition(window_width, window_height); // Testing purposes, starts all shapes in the center of screen.
        shape.setFillColor(sf::Color::Yellow);
        shape.setOutlineColor(sf::Color::White);
        shape.setOutlineThickness(1);
        shape.setRadius(bodiesSize);

        // Adding the body to the flock and adding the shapes to the vector<sf::CircleShape>
        flock.addBody(b);
        shapes.push_back(shape);
    }
    while (window.isOpen()) {
        HandleInput();
        Render();
    }
}

void Sim::HandleInput()
{
    sf::Event event;
    while (window.pollEvent(event)) {
        // "close requested" event: we close the window
        // Implemented alternate ways to close the window. (Pressing the escape, X, and BackSpace key also close the program.)
        if ((event.type == sf::Event::Closed) ||
            (event.type == sf::Event::KeyPressed &&
             event.key.code == sf::Keyboard::Escape) ||
            (event.type == sf::Event::KeyPressed &&
             event.key.code == sf::Keyboard::BackSpace) ||
            (event.type == sf::Event::KeyPressed &&
             event.key.code == sf::Keyboard::X))
             {
            window.close();
        }
    }
    // Check for mouse click, draws and adds bodies to flock if so.
    if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
        // Gets mouse coordinates, sets that as the location of the body and the shape
        sf::Vector2i mouseCoords = sf::Mouse::getPosition(window);
        Body b(mouseCoords.x, mouseCoords.y, false);
        sf::CircleShape shape(4);

        // Changing visual properties of newly created body
        shape.setPosition(mouseCoords.x, mouseCoords.y);
        shape.setOutlineColor(sf::Color::White);
        shape.setFillColor(sf::Color::White);
        shape.setOutlineColor(sf::Color::White);
        shape.setOutlineThickness(1);
        shape.setRadius(bodiesSize);

        // Adds newly created body and shape to their respective data structure
        flock.addBody(b);
        shapes.push_back(shape);

        // New Shape is drawn
        window.draw(shapes[shapes.size()-1]);
    }
}

void Sim::Render()
{
    window.clear();

    // Draws all of the bodies out, and applies functions that are needed to update.
    for (int i = 0; i < shapes.size(); i++) {
        window.draw(shapes[i]);

        //cout << "Body "<< i <<" Coordinates: (" << shapes[i].getPosition().x << ", " << shapes[i].getPosition().y << ")" << endl;
        //cout << "Body Code " << i << " Location: (" << flock.getBody(i).location.x << ", " << flock.getBody(i).location.y << ")" << endl;

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
    }

    // Applies the three rules to each body in the flock and changes them accordingly.
    flock.flocking();

    window.display();
}
