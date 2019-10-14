/*
 * Swarmbot.h
 *
 *  Created on: May 29, 2019
 *      Author: Trev
 */

#ifndef SWARMBOT_H_
#define SWARMBOT_H_

#include <cstdlib>
#include <list>

class Swarmbot {
private:

	//world attributes
	int const empty = 0;
	int const food = 1;
	int const habitat = 2;
	int const human = 3;
	int const unexplored = 4;
	int const robot = 5;
	// personality constants
	int const generosity = 6;
	int const talking = 7
	int const gamma = 8;
	int const sightRng = 9;

	//map level
	int const envior = 0;
	int const robots = 1;
	int const humans = 2;


	int widthWorld;
	int lengthWorld;
	//[space wanted, Food wanted, habitat wanted, human infulence wanted, exploration wanted, communication, share resources, trust in static enviorment, vision range]
	double personality[9];

	//[space level, food level, habitat level, human influence level, exploration level]
	int health[5];

	//number of ticks of time lived
	int lifetime;

	//[x, y] on map
	int location[2];

	//known world Map
	int ***KnownMap;

	//map you can see
	int ***LocalMap

	//add in Qtable

	//robot is dead = true, robot is alive = false
	bool isDead;



	//int* findClosest(int type, int robot_enviorment);
	//int* findFurthest(int type, int robot_enviorment);
	//int* avgLocation(int type);
	//int* furtherLessdxdy(int type);

	//move to desired location
	void move();

	//age with time
	void aging();

	//share resources
	void share(Swarmbot otherbot, int resourceType, int amount);

	//Communicate
	void communicate(Swarmbot otherbot);

	//look around (fill local map)
	void lookAround(int localWorld[sightRng][sightRng][3]);

	//die and reset
	void die();



public:
	//constructors / deconstructors
	Swarmbot(){};
	//			world width, world length, location x, location y, personality of the robot
	Swarmbot(int width, int length, int x, int y, int* personal[7]);
	virtual ~Swarmbot();

	//udate the map at a specific location, what map you are updated, and the new value
	void updateMap(int x, int y, int level, int value);

	int* getHealth();
	int* getLocation();
	void setLocation(int x, int y);

	//progress through time
	void exist();

	int getMap(int x, int y, int level);
	void printKnownWorld(int level);

};

#endif /* SWARMBOT_H_ */
