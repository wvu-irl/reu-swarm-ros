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

#include <math.h>
#include <swarm_simulation/Pvector.h>


//Pvector Functions from Pvector.h
//--------------------------------

#define PI 3.141592635

// Sets values of x and y for Pvector
void Pvector::set(float i, float o)
{
    x = i;
    y = o;
}

void Pvector::addVector(Pvector v)
{
    x += v.x;
    y += v.y;
}

// Adds to a Pvector by a constant number
void Pvector::addScalar(float s)
{
    x += s;
    y += s;
}

// Subtracts 2 vectors
void Pvector::subVector(Pvector v)
{
    x -= v.x;
    y -= v.y;
}

// Subtracts two vectors and returns the difference as a vector
Pvector Pvector::subTwoVector(Pvector v, Pvector v2)
{
    Pvector tmp;
    v.x -= v2.x;
    v.y -= v2.y;
    tmp.set(v.x, v.y);
    return tmp;
}

// Adds to a Pvector by a constant number
void Pvector::subScalar(float s)
{
    x -= s;
    y -= s;
}

// Multiplies 2 vectors
void Pvector::mulVector(Pvector v)
{
    x *= v.x;
    y *= v.y;
}

// Adds to a Pvector by a constant number
void Pvector::mulScalar(float s)
{
    x *= s;
    y *= s;
}

// Divides 2 vectors
void Pvector::divVector(Pvector v)
{
    x /= v.x;
    y /= v.y;
}

// Adds to a Pvector by a constant number
void Pvector::divScalar(float s)
{
    x /= s;
    y /= s;
}

void Pvector::limit(double max)
{
    double size = magnitude();

    if (size > max) {
        set(x / size, y / size);
    }
}

// Calculates the distance between the first Pvector and second Pvector
float Pvector::distance(Pvector v)
{
    float dx = x - v.x;
    float dy = y - v.y;
    float dist = sqrt(dx*dx + dy*dy);
    return dist;
}

// Calculates the dot product of a vector
float Pvector::dotProduct(Pvector v)
{
    float dot = x * v.x + y * v.y;
    return dot;
}

// Calculates magnitude of referenced object
float Pvector::magnitude()
{
    return sqrt(x*x + y*y);
}

void Pvector::setMagnitude(float x)
{
    normalize();
    mulScalar(x);
}

// Calculate the angle between Pvector 1 and Pvector 2
float Pvector::angleBetween(Pvector v)
{
    if (x == 0 && y == 0) return 0.0f;
    if (v.x == 0 && v.y == 0) return 0.0f;

    double dot = x * v.x + y * v.y;
    double v1mag = sqrt(x * x + y * y);
    double v2mag = sqrt(v.x * v.x + v.y * v.y);
    double amt = dot / (v1mag * v2mag); //Based of definition of dot product
    //dot product / product of magnitudes gives amt
    if (amt <= -1) {
        return PI;
    } else if (amt >= 1) {
        return 0;
    }
    float tmp = acos(amt);
    return tmp;
}

// normalize divides x and y by magnitude if it has a magnitude.
void Pvector::normalize()
{
    float m = magnitude();

    if (m > 0) {
        set(x / m, y / m);
    } else {
        set(x, y);
    }
}

// Creates and returns a copy of the Pvector used as a parameter
Pvector Pvector::copy(Pvector v)
{
    Pvector copy(v.x, v.y);
    return copy;
}
