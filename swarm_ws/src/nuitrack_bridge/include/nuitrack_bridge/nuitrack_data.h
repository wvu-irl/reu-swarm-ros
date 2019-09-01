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

/*
 * IMPORTANT:
 *  This file must use standard libraries only, such as cstdlib and Boost.
 *  Do NOT add any dependencies to ROS. This same header is used by the Nuitrack
 *  bridge UDP server, which cannot be compiled with ROS.
 *  Thanks :) nh
 */

#ifndef NUITRACK_DATA_H
#define NUITRACK_DATA_H

#include <string>

using std::string;

// Enumeration to define some types of gestures
enum class gestureType : char
{
    NONE = '\0',
    WAVING = (char)1,
    SWIPE_LEFT = (char)2,
    SWIPE_RIGHT = (char)3,
    SWIPE_UP = (char)4,
    SWIPE_DOWN = (char)5,
    PUSH = (char)6
};

// Simple struct to contain x, y, and z. Like a geometry_msgs::Point just with
//   less crazy typecasting and far simpler constructors.
typedef struct xyz
{
    xyz() //Default Constructor
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    xyz(double _x, double _y, double _z) //Alternate Constructor
    {
        x = _x;
        y = _y;
        z = _z;
    }

    double x;
    double y;
    double z;
} xyz;

// Struct to contain all relevant info for us. This struct is shared between
//   both halves of the bridge (UDP server and ROS client), so changes must be
//   addressed in both files.
typedef struct nuiData
{
    nuiData() //Default Constructor
    {
        leftFound = false;
        rightFound = false;
        gestureData = gestureType::NONE;
        leftHand = xyz();
        leftWrist = xyz();
        rightHand = xyz();
        rightWrist = xyz();
        confLH = 0.0;
        confLW = 0.0;
        confRH = 0.0;
        confRW = 0.0;
        leftClick = false;
        rightClick = false;
    }

    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gT,
            xyz _lH, xyz _lW, xyz _rH, xyz _rW,
            double _cLH, double _cLW, double _cRH, double _cRW,
            bool _lC, bool _rC) //Alternate Constructor
    {
        gestureFound = _gF;
        leftFound = _lF;
        rightFound = _rF;
        gestureData = _gT;
        leftHand = _lH;
        leftWrist = _lW;
        rightHand = _rH;
        rightWrist = _rW;
        confLH = _cLH;
        confLW = _cLW;
        confRH = _cRH;
        confRW = _cRW;
        leftClick = _lC;
        rightClick = _rC;
    }

    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gT,
            double _lHX, double _lHY, double _lHZ,
            double _lWX, double _lWY, double _lWZ,
            double _rHX, double _rHY, double _rHZ,
            double _rWX, double _rWY, double _rWZ,
            double _cLH, double _cLW, double _cRH, double _cRW,
            bool _lC, bool _rC) //Alternate Constructor
    {
        gestureFound = _gF;
        leftFound = _lF;
        rightFound = _rF;
        gestureData = _gT;
        leftHand = xyz(_lHX, _lHY, _lHZ);
        leftWrist = xyz(_lWX, _lWY, _lWZ);
        rightHand = xyz(_rHX, _rHY, _rHZ);
        rightWrist = xyz(_rWX, _rWY, _rWZ);
        confLH = _cLH;
        confLW = _cLW;
        confRH = _cRH;
        confRW = _cRW;
        leftClick = _lC;
        rightClick = _rC;
    }

    // Are gesture data, left hand data, and right hand data valid
    bool gestureFound, leftFound, rightFound;

    // Type of gesture
    gestureType gestureData;

    // Data for each point
    xyz leftHand, rightHand, leftWrist, rightWrist;

    // Confidence for each point
    double confLH, confRH, confLW, confRW;

    // "Clicks" are Nuitrack's way of knowing if a hand is closed
    bool leftClick, rightClick;
} nuiData;

#endif /* NUITRACK_DATA_H */
