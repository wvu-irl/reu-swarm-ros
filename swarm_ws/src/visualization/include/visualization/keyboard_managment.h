#ifndef KEY_MGR_H
#define KEY_MGR_H

#include <SFML/Graphics.hpp>
#include <ros/ros.h>

#include <wvu_swarm_std_msgs/map_levels.h>

#include <contour_node/level_description.h>

#include "visualization_settings.h"
#include "perspective_transform_gpu.h"

#define WORKING_LEVEL map_ns::TARGET

namespace interaction
{

sf::Vector2f getMouseCordinate(sf::Vector2f screen_loc, quadrilateral_t quad);

void keyEvent(sf::Event e);

void mousePressedEvent(sf::Event e);
void mouseReleasedEvent(sf::Event e);
void mouseMovedEvent(sf::Event e);

void scrollWheelMoved(sf::Event e);

void init(ros::Publisher *_add_pub, ros::Publisher *_rem_pub,
		ros::Publisher *_loc_pub, wvu_swarm_std_msgs::map_levels *_universe,
		quadrilateral_t _table);
void updateUni(wvu_swarm_std_msgs::map_levels *_universe);

} // interaction

#endif /* KEY_MGR_H */
