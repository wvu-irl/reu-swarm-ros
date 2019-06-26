/*
 * Terrain.cpp
 *
 *  Created on: May 29, 2019
 *      Author: Trevs
 */

#include "Terrain.h"
#include <random>
#include "Swarmbot.h"
#include <array>
#include <vector>

int height = 10;
int width = 10;
int tMap[10][10][2];



Terrain::Terrain(int w, int h) {
	height = h;
	width = w;

	//make a terrain of random stuff and no robots map[x][y][robot?]
		for (int x = 0; x < width; x++) {
			for (int y=0; y < height; y++) {

				int enviorment = rand() % 3;
					//int enviorment = 0;

				tMap[x][y][0] = enviorment;
				tMap[x][y][1] = 0;
			}

		}
		//tMap[0][5][0] = 2;
		//tMap[5][0][0] = 1;

}

Terrain::~Terrain() {
	// TODO Auto-generated destructor stub
}

int* Terrain::addRobot() {
	coord[0] = -1;
		coord[1]= - 1;

		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				if (tMap[x][y][1] != 1) {
					tMap[x][y][1] = 1;
					coord[0] = x;
					coord[1] = y;
					return coord;
				}

			}

		}
		return coord;
}

int Terrain::getPoint(int x, int y) {
	return tMap[x][y][0];
}

int Terrain::isRobotThere(int x, int y) {
	if (tMap[x][y][1] == 1)
			return 1;
		else
			return 0;
}

void Terrain::updateTMap(std::vector <Swarmbot> bots)
{
	//wipe map of robots
	int y = 0;
	for (int x = 0; x < width; x++) {
		for (y = 0; y < height; y++) {

			tMap[x][y][1] = 0;
		}

	}

	//add robots at new location
	//for (Swarmbot* botID : bots) {
	//	Swarmbot bot = *botID;
	for(Swarmbot bot : bots){
		int location[2];
		location[0]= *bot.getLocation();
		location[1]=*(bot.getLocation()+1);
		int x = location[0];
		int y = location[1];

		tMap[x][y][1] = 1;
	}

}
