#pragma once
#include <math.h>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <functional>

const int HEIGHT = 400;
const int WIDTH = HEIGHT * 2;

const int robot_size = 10;
const int speed = 2;

struct model
{
    
};

void *getBot(int i);
int getSwarmSize();

// returns the angle between two points
double anglebetween(double x0, double y0, double x1, double y1)
{
    if (x0 == x1)
    {
        if (y0 > y1)
        {
            return -M_PI_2;
        }
        else
        {
            return M_PI_2;
        }
    }
    return atan((y1 - y0) / (x1 - x0)) + (x1 < x0 ? M_PI : 0);
}

// returns the distance between two points
double pointdist(double x0, double y0, double x1, double y1)
{
    return sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
}

// Internal robot simulacra

class HolonomicBot
{
private:
    sf::RectangleShape rect;
    sf::VertexArray targetline;

    sf::Rect<int> bounds;

    std::function<sf::Vector2f(struct model)> rule;

    struct model rob_model;

public:
    int x, y, tx, ty;

    HolonomicBot(int initialX, int initialY, std::function<sf::Vector2f(struct model)> rule_callback)
    {
        this->x = initialX;
        this->y = initialY;
        this->tx = this->x;
        this->ty = this->y;

        this->rule = rule_callback;

        // rendering details
        rect.setFillColor(sf::Color::Blue);
        rect.setSize(sf::Vector2f((float)robot_size, (float)robot_size));
        rect.setPosition(sf::Vector2f((float)(this->x - robot_size / 2), (float)(this->y - robot_size / 2)));
        rect.setOutlineColor(sf::Color::White);
        rect.setOutlineThickness(1);

        targetline = sf::VertexArray(sf::LinesStrip, 2);
        targetline[0].color = sf::Color::Red;
        targetline[0].position = sf::Vector2f(this->x, this->y);
        targetline[1].color = sf::Color::Red;
        targetline[1].position = sf::Vector2f(this->tx, this->ty);

        bounds.width = robot_size;
        bounds.height = robot_size;

        
    }

    bool onTarget()
    {
        return x == tx && y == ty;
    }

    // Computation side
    void tick()
    {
        // computing target
        sf::Vector2f vel = rule(this->rob_model);
        
        // telling the robot what to do
        sf::Vector2i v = ardtick(this->x, this->y, this->tx, this->ty);

        // Simulation portion only
        this->x += v.x;
        this->y += v.y;
    }

    // On the robot
    virtual sf::Vector2i ardtick(int x, int y, int xt, int yt)
    {
        bounds.left = this->x - robot_size / 2;
        bounds.top = this->y - robot_size / 2;

        // speed regulation
        int useableSpeed = speed;
        if (pointdist(x, y, xt, yt) < speed)
        {
            useableSpeed = 1;
        }

        // Holonomic motion
        double ang = anglebetween(x, y, xt, yt);

        if (x >= WIDTH - robot_size / 2)
        {
            this->x = WIDTH - robot_size / 2;
        }
        else if (x <= robot_size / 2)
        {
            this->x = robot_size / 2;
        }

        if (y >= HEIGHT - robot_size / 2)
        {
            this->y = HEIGHT - robot_size / 2;
        }
        else if (y <= robot_size / 2)
        {
            this->y = robot_size / 2;
        }

        for (size_t i = 0; i < getSwarmSize(); i++)
        {
            if (bounds.intersects(((HolonomicBot *)getBot(i))->getBounds()))
            {
                
            }
        }

        // for simulation
        return sf::Vector2i(useableSpeed * cos(ang), useableSpeed * sin(ang));
    }

    void render(sf::RenderWindow *window)
    {
        rect.setPosition(sf::Vector2f((float)(this->x - robot_size / 2), (float)(this->y - robot_size / 2)));
        targetline[0].position = sf::Vector2f(this->x, this->y);
        targetline[1].position = sf::Vector2f(this->tx, this->ty);
        window->draw(rect);
        window->draw(targetline);
    }

    sf::Rect<int> getBounds()
    {
        return bounds;
    }
};

std::vector<HolonomicBot> swarm;
void *getBot(int i)
{
    return &(swarm.at(i));
}

int getSwarmSize()
{
    return swarm.size();
}