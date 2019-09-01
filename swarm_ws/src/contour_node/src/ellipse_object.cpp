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

#include <contour_node/ellipse_object.h>

ellipseObject::ellipseObject(void)
{
}

ellipseObject::ellipseObject(std::pair<double, double> _orig, std::string _name,
		std::pair<double, double> _radii, double _theta, levelType _lvl) :
		levelObject(_orig, _name, _lvl), radii(_radii), theta(_theta)
{
}

ellipseObject::ellipseObject(double _xorg, double _yorg, std::string _name,
		double _xrad, double _yrad, double _theta, levelType _lvl) :
		levelObject(_xorg, _yorg, _name, _lvl)
{
	radii = std::pair<double, double>(_xrad, _yrad);
	theta = _theta;
}

ellipseObject::ellipseObject(wvu_swarm_std_msgs::ellipse _msg,
		std::string _name, levelType _lvl) :
		levelObject(_msg.offset_x, _msg.offset_y, _name, _lvl), theta(
				_msg.theta_offset), radii(
				std::pair<double, double>(_msg.x_rad, _msg.y_rad))
{
}

ellipseObject::~ellipseObject(void)
{
}

void ellipseObject::nuiManipulate(double _x, double _y, double _z)
{
	// TODO: change the ellipse as the hand moves
}

std::pair<double, double> ellipseObject::getRadii(void)
{
	return radii;
}
double ellipseObject::getTheta(void)
{
	return theta;
}

wvu_swarm_std_msgs::ellipse ellipseObject::getEllipseMessage(void)
{
	wvu_swarm_std_msgs::ellipse el;
	el.x_rad = radii.first;
	el.y_rad = radii.second;
	el.offset_x = origin.first;
	el.offset_y = origin.second;
	el.theta_offset = theta;
	return el;
}
