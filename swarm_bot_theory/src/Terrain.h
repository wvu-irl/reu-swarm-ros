/*
 * Terrain.h
 *
 *  Created on: May 29, 2019
 *      Author: Trevs
 */

#ifndef TERRAIN_H_
#define TERRAIN_H_
#include "Swarmbot.h"
#include <vector>
#include <array>
#include <cstdlib>
#include <list>

class Terrain {

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


	int ***world;
	std::vector<Swarmbot> bots;
	int twidth;
	int tlength;
	int numbots;

public:

	Terrain(int w, int h, int r);
	virtual ~Terrain();

	void addRobot();
	int getWorld(int x, int y, int level);
	bool isRobotThere(int x, int y);
	void setHumanInfluence(std::list<int*> coords);
	void updateWorld();
	void printMap(int level);
	void testingFunction();

	int getTlength() const {
		return tlength;
	}

	int getTwidth() const {
		return twidth;
	}
};

#endif /* TERRAIN_H_ */
