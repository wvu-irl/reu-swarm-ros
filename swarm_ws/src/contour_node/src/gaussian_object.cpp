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

#include <contour_node/gaussian_object.h>

gaussianObject::gaussianObject(void)
{
}

gaussianObject::gaussianObject(std::pair<double, double> _orig,
		std::string _name, std::pair<double, double> _radii, double _theta,
		double _ampl, levelType _lvl) :
		ellipseObject(_orig, _name, _radii, _theta, _lvl), amplitude(_ampl)
{
}

gaussianObject::gaussianObject(double _xorg, double _yorg, std::string _name,
		double _xrad, double _yrad, double _theta, double _ampl, levelType _lvl) :
		ellipseObject(_xorg, _yorg, _name, _xrad, _yrad, _theta, _lvl), amplitude(
				_ampl)
{
}

gaussianObject::gaussianObject(wvu_swarm_std_msgs::obstacle message) :
		ellipseObject(message.characteristic.ellipse, message.characteristic.name,
				(levelType) message.level), amplitude(message.characteristic.amplitude)
{
}

gaussianObject::~gaussianObject(void)
{
}

void gaussianObject::nuiManipulate(double _x, double _y, double _z)
{
	// If hand's height moved, change amplitude
	if (_z > 2.0)
		amplitude += 1.5;
	else if (_z < -2.0)
		amplitude -= 1.5;
	else if (abs(_z) > 0.5)
		amplitude += (_z - 0.5);

	// If hand's location moved, move the origin
	if (_x > 1.0)
		origin.first += 0.5;
	else if (_x < -1.0)
		origin.first -= 0.5;
	else if (abs(_x) > 0.1)
		origin.first += (_x);
	if (_y > 1.0)
		origin.second += 0.5;
	else if (_y < -1.0)
		origin.second -= 0.5;
	else if (abs(_y) > 0.1)
		origin.second += (_y);
}

double gaussianObject::getAmplitude(void)
{
	return amplitude;
}

wvu_swarm_std_msgs::obstacle gaussianObject::getGaussianMessage(void)
{
	wvu_swarm_std_msgs::obstacle msg;
	wvu_swarm_std_msgs::gaussian gaus;
	gaus.ellipse = getEllipseMessage();
	gaus.amplitude = amplitude;
	gaus.name = name;
	gaus.selected = isSelected();

	msg.characteristic = gaus;
	msg.level = (int)level;
	return msg;
}

