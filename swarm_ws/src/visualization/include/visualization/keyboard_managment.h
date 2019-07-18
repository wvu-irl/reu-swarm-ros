#ifndef KEY_MGR_H
#define KEY_MGR_H

#include <SFML/Graphics.hpp>
#include <ros/ros.h>

#include <contour_node/universe_object.h>
#include <wvu_swarm_std_msgs/map_levels.h>

#include "visualization_settings.h"
#include "perspective_transform_gpu.h"

namespace interaction
{

Universe universe;
ros::Publisher add_pub;

sf::Vector2f getMouseCordinate(sf::Vector2f screen_loc, quadrilateral_t quad);

void keyEvent(sf::Event e);

void mousePressedEvent(sf::Event e);
void mouseReleasedEvent(sf::Event e);
void mouseMovedEvent(sf::Event e);

} // interaction

#endif /* KEY_MGR_H */
