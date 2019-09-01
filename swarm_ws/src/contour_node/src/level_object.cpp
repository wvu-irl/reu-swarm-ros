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

#include <contour_node/level_object.h>

levelObject::levelObject(void)
{
    origin.first = 0.0;
    origin.second = 0.0;
    name = "";
    selected = false;
}

levelObject::levelObject(std::pair<double, double> _orig, std::string _name, levelType _lvl)
    : origin(_orig), name(_name), selected(false), level(_lvl) {
}

levelObject::levelObject(double _xorg, double _yorg, std::string _name, levelType _lvl) : level(_lvl)
{
    origin = std::pair<double, double>(_xorg, _yorg);
    name = _name;
}

levelObject::~levelObject(void) {}

std::pair<double, double> levelObject::getOrigin(void) {return origin;}

void levelObject::setName(std::string _s) {name = _s;}
std::string levelObject::getName() {return name;}

void levelObject::select(void) {selected = true;}
void levelObject::deselect(void) {selected = false;}
bool levelObject::isSelected(void) {return selected;}

levelType levelObject::getLevel(void) {return level;}
void levelObject::setLevel(levelType _lvl) {level = _lvl;}
