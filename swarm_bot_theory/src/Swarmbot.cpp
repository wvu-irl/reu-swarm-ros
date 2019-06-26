/*
 * Swarmbot.cpp
 *
 *  Created on: May 29, 2019
 *      Author: Trevs
 */

#include "Swarmbot.h"
#include <iostream>
#include <cmath>
#include "Terrain.h"
#include <cstdlib>


int foodLevel;
int habitatLevel;
int location[2];
int old;
int sMap[10][10][2];
int swidth = 10;
int sheight = 10;

Swarmbot::Swarmbot(int xy[2], int m[10][10][2]) {
		foodLevel = 100;
		habitatLevel = 100;
		for(int x =0; x < 10; x++){
			for(int y = 0; y < 10; y++){
				sMap[x][y][0] = m[x][y][0];
				sMap[x][y][1] = m[x][y][1];
			}
		}

		location[0] = xy[0];
		location[1] = xy[1];
		old = 0;
}


Swarmbot::~Swarmbot() {
	// TODO Auto-generated destructor stub
}

int* Swarmbot::findFood() {

		coord[0] = -1;
		coord[1] = -1;
		for (int x = 0; x < 10; x++) {
			for (int y = 0; y < 10; y++) {

				if (sMap[x][y][0] == 2 && sMap[x][y][1] != 1) {
					coord[0] = x;
					coord[1] = y;
					return coord;
				}
			}

		}
		return coord;
}


int* Swarmbot::findHabitat() {
		coord[0] = -1;
		coord[1] = -1;
		for (int x = 0; x < 10; x++) {
			for (int y = 0; y < 10; y++) {

				if (sMap[x][y][0] == 1 && sMap[x][y][1] != 1) {
					coord[0] = x;
					coord[1] = y;
					return coord;
				}
			}

		}
		return coord;
}

void Swarmbot::move() {
	//food vector {mag, x, y}
		int foodVector[3];
		foodVector[0] = 100 - foodLevel;
		if(foodVector[0] < 0){
			foodVector[0] = 0;
		}
		int* foodLocation = this->findFood();
		foodVector[1] = *(foodLocation);
		foodVector[2] = *(foodLocation+1);
		double foodDistMag = 1;
		if(foodVector[1] == -1 || foodVector[2] == -1){
			foodVector[1] = 0;
			foodVector[2] = 0;
		}
		else{
			foodVector[1] = foodVector[1] - location[0];
			foodVector[2] = foodVector[2] - location[1];
			float foodx = foodVector[1];
			float foody = foodVector[2];
			foodDistMag = sqrt(pow(foodx, 2)+(pow(foody,2)));
			}


		//habitat vector {mag, x, y}
		int habitatVector[3];
		habitatVector[0] = 100 - habitatLevel;
		if(habitatVector[0] < 0){
			habitatVector[0] = 0;
		}
		int* habitatLocation = this->findHabitat();
		habitatVector[1] = *(habitatLocation);
		habitatVector[2] = *(habitatLocation+1);
		double habitatDistMag = 1;
		if(habitatVector[1] == -1 || habitatVector[2] == -1){
				habitatVector[1] = 0;
				habitatVector[2] = 0;
			}
		else{
			habitatVector[1] = habitatVector[1] - location[0];
			habitatVector[2] = habitatVector[2] - location[1];
			float habitatx = habitatVector[1];
			float habitaty = habitatVector[2];
			habitatDistMag = sqrt(pow(habitatx,2) + pow(habitaty,2));
		}



		double xTot = (foodVector[0] * (foodVector[1]/foodDistMag))  + (habitatVector[0] * (habitatVector[1]/habitatDistMag));
		double yTot = (foodVector[0] * (foodVector[2]/foodDistMag)) + (habitatVector[0] * (habitatVector[2]/habitatDistMag));
		long xRound = xTot;
		long yRound = yTot;
		long foodDistRound = foodDistMag;
		long habitatDistRound = habitatDistMag;
		std::cout<<"food dist= "<<foodDistRound<< " x= "<<foodVector[1] <<" y= "<< foodVector[2]<< std::endl;
		std::cout<<"habitat dist= " << habitatDistRound<<" x= "<< habitatVector[1]<<" y= "<<habitatVector[2]<<std::endl;
		std::cout<<"x tot= "<< xRound << " y tot " << yRound<< std::endl;
		sMap[location[0]][location[1]][1] = 0;

		if (abs(xTot) > abs(yTot)) {
			if (xTot > 0 && location[0] < swidth -1) {
				location[0]++;
			}
			else if (xTot < 0 && location[0] > 0)
			{
				location[0]--;
			}
		}
		else {

			if (yTot > 0  && location[1] < sheight -1) {
				location[1]++;
			}
			else if (yTot < 0 && location[1] > 0)
			{
				location[1]--;
			}
		}
		sMap[location[0]][location[1]][1] = 1;
}


void Swarmbot::age() {
	old++;
		foodLevel = foodLevel - 10;
		habitatLevel = habitatLevel - 5;
		if (sMap[location[0]][location[1]][0] == 1) {
			habitatLevel = habitatLevel + 10;
		}

		if (sMap[location[0]][location[1]][0] == 2) {
			foodLevel = foodLevel + 50;
		}
}

void Swarmbot::die(){
	sMap[location[0]][location[1]][0] = 0;
	if(foodLevel == 0 || habitatLevel == 0){
		location[0] = -1;
		location[1] = -1;
	}
}

void Swarmbot::printCoord(){
	std::cout<< "x = "<<location[0]<< "  y = " << location[1]<<std::endl;
}

void Swarmbot::updateSMap(int m[10][10][2]){
	for(int x =0; x < 10; x++){
				for(int y = 0; y < 10; y++){
					sMap[x][y][0] = m[x][y][0];
					sMap[x][y][1] = m[x][y][1];
				}
			}
}



