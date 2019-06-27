/*
 * Terrain.h
 *
 *  Created on: May 29, 2019
 *      Author: Trevs
 */
//test
#ifndef TERRAIN_H_
#define TERRAIN_H_
#include "Swarmbot.h"
#include <vector>

class Terrain {

private:
	int height;
	int width;
	int coord[2];

public:

	int tMap[10][10][2]; //????
	Terrain(int w, int h);
	virtual ~Terrain();
	int* addRobot();
	int getPoint(int x, int y);
	int isRobotThere(int x, int y);
	void updateTMap(std::vector<Swarmbot> bots);
};

#endif /* TERRAIN_H_ */
