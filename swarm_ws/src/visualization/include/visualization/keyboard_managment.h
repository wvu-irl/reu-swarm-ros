/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

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
