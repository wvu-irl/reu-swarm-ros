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

#include <iostream>

using namespace std;

#ifndef PVECTOR_H_
#define PVECTOR_H_

// The Pvector class implements vectors have both a magnitude and a direction.
// Pvectors is for implementing movement and the three Body rules -- cohesion, separation, and matching velocity
// through the use of acceleration, force, and velocity vectors.

class Pvector {

public:
    float x;
    float y;

    //Constructors
    Pvector() {}

    Pvector(float xComp, float yComp)
    {
        x = xComp;
        y = yComp;
    }

    //Mutator Functions
    void set(float x, float y);

    //Scalar functions scale a vector by a float
    void addVector(Pvector v);
    void addScalar(float x);

    void subVector(Pvector v);
    Pvector subTwoVector(Pvector v, Pvector v2);
    void subScalar(float x);

    void mulVector(Pvector v);
    void mulScalar(float x);

    void divVector(Pvector v);
    void divScalar(float x);

    void limit(double max);

    //Calculating Functions
    float distance(Pvector v);
    float dotProduct(Pvector v);
    float magnitude();
    void setMagnitude(float x);
    float angleBetween(Pvector v);
    void normalize();

    Pvector copy(Pvector v);
};

#endif
