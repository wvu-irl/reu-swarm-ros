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
