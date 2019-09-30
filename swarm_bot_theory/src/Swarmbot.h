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
	int const food = 1;
	int const habitat = 2;
	int const human = 3;
	int const unexplored = 4;
	int const robot = 5;
	int const envior = 0;
	int const robots = 1;
	int const humans = 2;

	int width;
	int length;
	double personality[5];
	int health[5];
	int lifetime;
	int location[2];
	int ***map;
	bool isDead;



	int* findClosest(int type, int robot_enviorment);
	int* findFurthest(int type, int robot_enviorment);
	int* avgLocation(int type);
	int* furtherLessdxdy(int type);
	void move();
	void aging();
	void die();



public:
	Swarmbot(){};
	Swarmbot(int width, int length, int x, int y);
	virtual ~Swarmbot();
	void updateMap(int x, int y, int level, int value);
	int* getHealth();
	int* getLocation();
	void setLocation(int x, int y);
	void exist();
	int getMap(int x, int y, int level);
	void knownWorld(int level);
	void printValues();
	int** Values;

};

#endif /* SWARMBOT_H_ */
