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

#ifndef UNIVERSE_H
#define UNIVERSE_H

#include <iostream>
#include <functional>

#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include "level_object.h"

/**
 * Class for managing all the functions that the robots need to use
 *
 *	- Uses the levels that are standardized in the level descriptor header
 *	- Allows finding of functions by name and by location
 *	- Operators to add new functions
 *			- += operator allows addition of a levelObject pointer or an obstacle message
 *	- Operators for removing functions
 *			- -= works for removing a function by name or location
 *
 */
class Universe
{
private:
	// container for all map data
	wvu_swarm_std_msgs::map_levels overall_map;

public:
	// default constructor
	Universe();

	// Constructor to create a map from a map_levels message
	Universe(wvu_swarm_std_msgs::map_levels _map);

	/**
	 * Adds an equation from an obstacle message
	 *
	 * @param equ is an obstacle message to be added to the collection of functions
	 */
	void addEquation(wvu_swarm_std_msgs::obstacle equ);
	/**
	 * Overload of addEquation (See addEquation(wvu_swarm_std_msgs::obstacle equ))
	 *
	 * @param equ is a levelObject pointer to be added to the set of equaitons
	 * 						- Currently the equ has to be the sub-type of a gaussianObject
	 */
	void addEquation(levelObject *equ);

	/**
	 * Finds the total number of equations in the entire collection
	 * 		not just for one map level, it is all the equations
	 *
	 * 	@returns the number of Gaussians in the universe
	 */
	size_t numEquaitons();

	/**
	 * Searches the Universe for a gaussian by the given name
	 *
	 * @param std::string a key to search for in the universe
	 *
	 * @returns a gaussian pointer to the function if found
	 * 					if not found NULL is returned
	 */
	wvu_swarm_std_msgs::gaussian* findMsgByName(std::string);

	/**
	 * Searches the Universe for a gaussian with an ellipse origin that is specified
	 *
	 * @param std::pair<double, double> a x,y pair that discribes the origin of the gaussian's ellipse
	 *
	 * @returns the gaussian pointer to the function if found
	 * 					if not found NULL is returned
	 *
	 */
	wvu_swarm_std_msgs::gaussian* findMsgByLocation(std::pair<double, double>);

	/**
	 * Searches the Universe for a levelObject by the given name
	 *
	 * @param std::string a key to search for in the universe
	 *
	 * @returns a gaussian pointer to the function if found
	 * 					if not found NULL is returned
	 */
	levelObject* findByName(std::string);

	/**
	 * Searches the Universe for a levelObject with an ellipse origin that is specified
	 *
	 * @param std::pair<double, double> a x,y pair that discribes the origin of the gaussian's ellipse
	 *
	 * @returns the gaussian pointer to the function if found
	 * 					if not found NULL is returned
	 *
	 */
	levelObject* findByLocation(std::pair<double, double>);
    
    /** TODO: THIS **/
    levelObject* findWithinRadius(std::pair<double, double> loc, double radius);

	/**
	 * Gets a reference to the map that is able to be published in mapping.cpp
	 *
	 * @returns a map_levels message to be published into ROS
	 */
	wvu_swarm_std_msgs::map_levels& getPublishable();

	/**
	 * Iterates through all the objects
	 *
	 * @param funk lambda function that takes a gaussian and a level and does something with them
	 */
	void foreach(std::function<void(wvu_swarm_std_msgs::gaussian*, int)> funk);

	///////////////////////////////////////////////////////////////////////////
	//                             Operators                                 //
	///////////////////////////////////////////////////////////////////////////

	// cast to map_levels message
	explicit operator wvu_swarm_std_msgs::map_levels&();

	// adding new functions
	friend void operator+=(Universe&, levelObject*);
	friend void operator+=(Universe&, wvu_swarm_std_msgs::obstacle&);

	// remove functions
	friend void operator-=(Universe&, std::string);
	friend void operator-=(Universe&, std::pair<double, double>);

	// print operator
	friend std::ostream& operator<<(std::ostream&, Universe&);
};

#endif
