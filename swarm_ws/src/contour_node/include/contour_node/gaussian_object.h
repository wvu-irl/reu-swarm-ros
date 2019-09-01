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

#ifndef GAUSSIANOBJECT_H
#define GAUSSIANOBJECT_H

#include "ellipse_object.h"
#include "level_description.h"
#include <wvu_swarm_std_msgs/obstacle.h>

class gaussianObject: public ellipseObject
{
public:
	gaussianObject(void);
	gaussianObject(std::pair<double, double> _orig, std::string _name,
			std::pair<double, double> radii, double _theta, double _ampl,
			levelType _lvl);
	gaussianObject(double _xorg, double _yorg, std::string _name, double _xrad,
			double _yrad, double _theta, double _ampl, levelType _lvl);

	gaussianObject(wvu_swarm_std_msgs::obstacle message);
	virtual ~gaussianObject(void);

	void nuiManipulate(double _x, double _y, double _z);

	double getAmplitude(void);

	wvu_swarm_std_msgs::obstacle getGaussianMessage(void);

protected:
	double amplitude;
};

#endif /* GAUSSIANOBJECT_H */

