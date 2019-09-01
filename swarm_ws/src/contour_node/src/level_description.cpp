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

#include <contour_node/level_description.h>
#include <math.h>
#include <iostream>
double map_ns::calculate(wvu_swarm_std_msgs::map_level ml,
		wvu_swarm_std_msgs::vicon_point loc)
{
	double rx = loc.x;
	double ry = loc.y;
	double z = 0;
	for (size_t i = 0; i < ml.functions.size(); i++)
	{
		wvu_swarm_std_msgs::gaussian curr_eq = ml.functions[i];
                double x = rx - curr_eq.ellipse.offset_x;
                double y = ry - curr_eq.ellipse.offset_y;

                double theta = x == 0 ? (y > 0 ? M_PI_2 : -M_PI_2) : (atan(y/x) + (y < 0 ? M_PI : 0));
                double r = sqrt(x*x + y*y);

                double a = curr_eq.ellipse.x_rad;
                double b = curr_eq.ellipse.y_rad;

                double x_app = r * cos(theta + curr_eq.ellipse.theta_offset);
                double y_app = r * sin(theta + curr_eq.ellipse.theta_offset);

                double re = a != 0 && b != 0 ? sqrt(a * a * x_app * x_app + y_app * y_app * b * b) / (a * b) : 10000;

		z += curr_eq.amplitude * pow(M_E, (-re * re) / 2.0);
	}

	return z;
}

wvu_swarm_std_msgs::map_level map_ns::combineLevels(
		wvu_swarm_std_msgs::map_level a, wvu_swarm_std_msgs::map_level b)
{
	wvu_swarm_std_msgs::map_level n_lev;
	for (size_t i = 0; i < a.functions.size(); i++)
		n_lev.functions.push_back(a.functions.at(i));

	for (size_t i = 0; i < b.functions.size(); i++)
		n_lev.functions.push_back(b.functions.at(i));

	return n_lev;
}
