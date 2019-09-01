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

#ifndef ELLIPSEOBJECT_H
#define ELLIPSEOBJECT_H

#include "level_object.h"
#include <wvu_swarm_std_msgs/ellipse.h>

class ellipseObject: public levelObject
{
public:
	ellipseObject(void);
	ellipseObject(std::pair<double, double> _orig, std::string _name,
			std::pair<double, double> _radii, double _theta, levelType _lvl);
	ellipseObject(double _xorg, double _yorg, std::string _name, double _xrad,
			double _yrad, double _theta, levelType _lvl);
	ellipseObject(wvu_swarm_std_msgs::ellipse _msg, std::string _name,
			levelType _lvl);
	virtual ~ellipseObject(void);

	void nuiManipulate(double _x, double _y, double _z);

	std::pair<double, double> getRadii(void);
	double getTheta(void);

	wvu_swarm_std_msgs::ellipse getEllipseMessage(void);

protected:
	std::pair<double, double> radii; // x, y
	double theta;
};

#endif /* ELLIPSEOBJECT_H */

