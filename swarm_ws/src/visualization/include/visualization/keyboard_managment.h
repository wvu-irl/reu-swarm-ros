#ifndef KEY_MGR_H
#define KEY_MGR_H

#include <SFML/Graphics.hpp>
#include <ros/ros.h>

#include <wvu_swarm_std_msgs/map_levels.h>

#include <contour_node/level_description.h>

#include "visualization_settings.h"
#include "perspective_transform_gpu.h"

namespace interaction
{

ros::Publisher add_pub, rem_pub, loc_pub;
wvu_swarm_std_msgs::map_levels *universe;
quadrilateral_t table;

sf::Vector2f getMouseCordinate(sf::Vector2f screen_loc, quadrilateral_t quad);

void keyEvent(sf::Event e);

void mousePressedEvent(sf::Event e);
void mouseReleasedEvent(sf::Event e);
void mouseMovedEvent(sf::Event e);

void scrollWheelMoved(sf::Event e);

} // interaction

#endif /* KEY_MGR_H */
