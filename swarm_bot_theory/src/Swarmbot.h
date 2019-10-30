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

	int const empty = 0;
	// personality constants
	int const talking = 0;
	int const food = 1;
	int const habitat = 2;
	int const human = 3;
	int const unexplored = 4;
	int const robot = 5;
	int const generosity = 6;
	int const gamma = 7;
	int const sightRng = 8;

	//map level
	int const foods = 0;
	int const habitats = 1;
	int const robots = 2;
	int const humans = 3;
	int const unknowns = 4;


	int widthWorld;
	int lengthWorld;
	//[communication, Food wanted, habitat wanted, human infulence wanted, exploration wanted, share resources, trust in static enviorment, vision range]
	double personality[8];

	//[communication level, food level, habitat level, human influence level, exploration level]
	int health[5];

	//number of ticks of time lived
	int lifetime;

	//[x, y] on map
	int location[2];

	//known world Map, [food, habitat, robots, human, unknown]
	double *****KnownMap;

	//map you can see
	double *****LocalMap;

	//add in Qtable

	//robot is dead = true, robot is alive = false
	bool isDead;



	//int* findClosest(int type, int robot_enviorment);
	//int* findFurthest(int type, int robot_enviorment);
	//int* avgLocation(int type);
	//int* furtherLessdxdy(int type);

	double* resultantResourceVector(int resource);

	//move to desired location
	void move();

	// get required resource
	void getResource(int resource);

	//age with time
	void aging();

	//share resources
	void share(Swarmbot otherbot, int resourceType, int amount);

	//Communicate
	void communicate(Swarmbot otherbot);

	//look around (fill local map)
	void lookAround(double localWorld[sightRng][sightRng][5]);

	//die and reset
	void die();



public:
	//constructors / deconstructors
	//Swarmbot(){};
	//			world width, world length, location x, location y, personality of the robot
	Swarmbot(int w, int l, int x, int y, int personal[9]);
	virtual ~Swarmbot();

	//update the map at a specific location, what map you are updated, and the new value
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
