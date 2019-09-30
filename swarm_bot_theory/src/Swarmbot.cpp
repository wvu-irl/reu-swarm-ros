/*
 * Swarmbot.cpp
 *
 *  Created on: May 29, 2019
 *      Author: Trevs
 */

#include "Swarmbot.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <list>


	int const empty = 0;
	int const food = 1;
	int const habitat = 2;
	int const human = 3;
	int const unexplored = 4;
	int const robot = 5;

	int const envior = 0;
	int const robots = 1;
	int const humans = 2;

	double personality[5];
	int health[5];
	int lifetime;
	int location[2];
	int ***map;
	int width;
	int length;
	bool isDead;
	int direction;


//-----------------------------------------------------------------constructors
Swarmbot::Swarmbot(int w, int l, int x, int y) {
	direction = 0;
	lifetime = 0;
	personality[0] = .20;
	personality[1] = .20;
	personality[2] = .20;
	personality[3] = .20;
	personality[4] = .20;
	health[0]= 100;
	health[1]= 100;
	health[2]= 100;
	health[3]= 100;
	health[4]= 100;
	isDead = false;
	location[0] = x;
	location[1] = y;
	width = w;
	length = l;
	map = new int**[width];
	for(int x = 0; x < width; x++){
		map[x] = new int*[length];
		for(int y = 0; y < length; y++){
			map[x][y] = new int[3];
		}
	}
	Values = new int*[width];
	for(int x = 0; x < width; x++){
		Values[x] = new int[length];
	}

	for(int x = 0; x < width; x++){
		for(int y = 0; y < length; y++){
			map[x][y][envior] = unexplored;
			map[x][y][robots]= 0;
			map[x][y][humans] = 0;
			Values[x][y] = 0;
		}
	}

}


Swarmbot::~Swarmbot() {
}


//------------------------------------------------------finds
int smallestCoords[2] = {100000,100000};
int* Swarmbot::findClosest(int type,int robot_enviorment) {
	smallestCoords[0] = 100000;
	smallestCoords[1] = 100000;
	bool found = false;
	for(int x = 0; x < width; x++){
		for(int y=0; y < length; y++){
			if(map[x][y][robot_enviorment] == type){
				int dx = x-location[0];
				int dy = y-location[1];
				int dist = abs(dx)+abs(dy);
				if(dist < (abs(smallestCoords[0]-location[0])+abs(smallestCoords[1]-location[1])) && dist != 0 && map[x][y][robots] != robot){
					smallestCoords[0] = x;
					smallestCoords[1] = y;
					found = true;
				}

			}
		}
	}
	if(!found){
		smallestCoords[0] = -1;
		smallestCoords[1] = -1;
	}
	return smallestCoords;
}



int furthestCoords[2] = {0,0};
int* Swarmbot::findFurthest(int type, int robot_enviorment){

		return furthestCoords;

}


//----------------------------------------------------actions
void Swarmbot::move() {

	double dx = 0;
	double dy = 0;
	double vdx =0;
	double vdy = 0;
	int notFound = 0;
	int* coordPointer;


	for(int i = 0; i < 5; i++){
		bool foundOne = true;
		int x;
		int y;
		if(i == empty){
			int* avgLoc = furtherLessdxdy(robot);
			int rDx = *(avgLoc);
			int rDy = *(avgLoc+1);

			x = location[0] -rDx;
			y = location[1]-rDy;
			if(x ==location[0] && y == location[1]){
				foundOne = false;
				x = -1;     //keep convention
				y = -1;     //keep convention
			}


		}
		else if(i == human){
			int* avgLoc = findClosest(i,humans);
			x = *(avgLoc);
			y = *(avgLoc+1);
			if(x == -1 || y == -1){
				notFound++;
				foundOne = false;
			}
		}
		else{
			coordPointer = findClosest(i, envior);
			x = *coordPointer;
			y = *(coordPointer+1);
			if(x == -1 || y == -1){
				notFound++;
				foundOne = false;
			}
		}
		std::cout<<"x = "<< x << " y= "<< y<<std::endl;
		double dist = sqrt(pow(x - location[0],2) + pow(y - location[1],2));
		if(foundOne){

			if(i == unexplored && !(x == -1 && y == -1) && dist != 0){
				//movment vector
				double unexpX = personality[i] * (100-health[i]) * ((x-location[0])/dist);
				double unexpY = personality[i] * (100-health[i]) * ((y-location[1])/dist);
				double notFoundX = notFound * .20 * 100 *((x-location[0])/dist);
				double notFoundY = notFound * .20 * 100 *((y-location[1])/dist);
				dx = dx + unexpX + notFoundX;
				dy = dy+ unexpY+ notFoundY;
				//value vector
				double ValunexpX = personality[i] *100* ((x-location[0])/dist);
				double ValunexpY =   personality[i] *100* ((y-location[1])/dist);
				double ValnotFoundX = notFound * .20 * 100 *((x-location[0])/dist);
				double ValnotFoundY = notFound * .20 *100* ((y-location[1])/dist);
				vdx = vdx + ValunexpX + ValnotFoundX;
				vdy = vdy+ ValunexpY+ ValnotFoundY;

				}
			else if(dist != 0){
				//movement vector
				dx = dx + personality[i] * (100-health[i]) * ((x-location[0])/dist);
				dy = dy + personality[i] * (100-health[i]) * ((y-location[1])/dist);
				//value vector
				vdx = vdx + personality[i] *100*  ((x-location[0])/dist);
				vdy = vdy + personality[i] * 100* ((y-location[1])/dist);
			}
			std::cout<<"dx = "<< dx << " dy= "<< dy<<" not found "<< notFound<<std::endl;
		}

	}
		Values[location[0]][location[1]] = sqrt(pow(vdx,2) + pow(vdy,2));
		map[location[0]][location[1]][robots] = empty;
	if(abs(dx) > abs(dy)){
			if(dx > 0 && location[0]< width-1 && map[location[0] + 1][location[1]][robots] != robot){
				location[0]+= 1;
			}
			else if(dx < 0 && location[0]>0 && map[location[0] - 1][location[1]][robots] != robot){
				location[0] -= 1;
			}
		}
		else{
			if(dy > 0 && location[1] < length-1 && map[location[0]][location[1]+1][robots] != robot){
				location[1]+=1;
			}
			else if(dy < 0 && location[1] >0 && map[location[0]][location[1]-1][robots] != robot){
				location[1] -=1;
			}
		}
		map[location[0]][location[1]][robots] = robot;

}

void Swarmbot::aging() {

	//get resources
	int resource = map[location[0]][location[1]][envior];
	int onHuman = map[location[0]][location[1]][humans];

	if(onHuman == human){
		health[human] += 10;
	}
	if(resource == food){
		health[food] += 10;
	}
	else if(resource == habitat){
		health[habitat] += 10;
	}
	else if(resource == unexplored){
		health[unexplored] += 10;
	}


	//update space level --- further better
	int* SCoord = avgLocation(robot);
		int Sx = *(SCoord);
		int Sy = *(SCoord+1);
		//std::cout<<"space X = " << Sx << " Space Y = "<< Sy << std::endl;
		if((Sx == -1 || Sy == -1) && map[location[0]][location[1]][robots] != robot){
			health[empty] = 100; //----------may need a scaling factor
		}
		else{
			int dist = abs(Sx - location[0]) + abs(Sy - location[1]);
			health[empty] = health[empty] + (dist - 5);
		}

	//drop foodlevel ---- time based
	health[food] = health[food] - 1;
	//drop habitatlevel  --- time based
	health[habitat] = health[habitat] - 1;
	//update human level ---  closer better
	int* HCoord = avgLocation(human);
	int Hx = *(HCoord);
	int Hy = *(HCoord+1);
	if((Hx == -1 || Hy == -1) && map[location[0]][location[1]][humans] != human){
		health[human] = 100;
	}
	else{
		int dist = abs(Hx - location[0]) + abs(Hy- location[1]);
		health[human] = health[human] -  dist;
	}
	//drop newlevel   --- time based
	health[unexplored] = health[unexplored] - 1;
	lifetime++;

	for(int g = 0; g < 5; g++){
		if(health[g] < 0)
			health[g] = 0;
		else if(health[g]>100)
			health[g] = 100;
	}


}

void Swarmbot::die() {
	int d = 0;
	for(int j = 0; j < 5; j++){
		if(health[j] <= 0){
			isDead = true;
			d = j;
		}
	}
	if(isDead){
		std::cout<<"dead  #################################################################" <<std::endl;
		//respawn and increase weight to how you died
		bool broken = false;
	for(int x = 0; x < width; x++){
		for(int y = 0; y < length; y++){
			if(map[x][y][robots] != robot){
				location[0] = x;
				location[1] = y;
				personality[d] += .5;
				broken = true;
				break;
			}
		}
		if(broken){
			break;
		}
	}
	health[0]= 100;
	health[1]= 100;
	health[2]= 100;
	health[3]= 100;
	health[4]= 100;
	isDead = false;
	}

}


//--------------------------------------------------------
void Swarmbot::updateMap(int x, int y, int level, int value) {
	map[x][y][level] = value;

}

int* Swarmbot::getHealth() {
	return health;
}

int* Swarmbot::getLocation() {
	return location;
}
void Swarmbot::setLocation(int x, int y){
	location[0] = x;
	location[1] = y;
}
void Swarmbot::exist(){
	move();

	aging();

	die();


}

void Swarmbot::knownWorld(int level){
	for (int y = 0; y < length; y++)
	{
		for (int x = 0; x < width; x++)
			{
			//cout << 4;
			std::cout <<map[x][y][level]<< " ";
			}
			std::cout << std::endl;
			}
		std::cout << "---------------------------------------------------------------------------------" << std::endl;
}
void Swarmbot::printValues(){
	for (int y = 0; y < length; y++)
	{
		for (int x = 0; x < width; x++)
			{
			//cout << 4;
			std::cout <<Values[x][y]<< " ";
			}
			std::cout << std::endl;
			}
		std::cout << "---------------------------------------------------------------------------------" << std::endl;
}


int avgLoc[2] = {-1,-1};
int* Swarmbot::avgLocation(int type){
	avgLoc[0] = -1;
	avgLoc[1] = -1;
	int level;
	if(type == human){
		level = humans;
	}
	else if(type == robot){
		level = robots;
	}
	else{
		level = envior;
	}

		int Sumx = 0;
		int Sumy = 0;
		int numOfType = 0;
		for(int a = 0; a < width; a++){
			for(int b = 0; b < length; b++){
				if((map[a][b][level] == type) && !(a == location[0] && b == location[1])){
					Sumx += a;
					Sumy += b;
					numOfType++;
				}
			}
		}
		int avgX = -1;
		int avgY = -1;
		if(numOfType > 0){
			avgX = Sumx / numOfType;
			avgY = Sumy / numOfType;
		}
		avgLoc[0] = avgX;
		avgLoc[1] = avgY;

	return avgLoc;

}
int dxdy[2] = {0,0};
int* Swarmbot::furtherLessdxdy(int type){

	dxdy[0] = 0;
	dxdy[1] = 0;
	int level;
		if(type == human){
			level = humans;
		}
		else if(type == robot){
			level = robots;
		}
		else{
			level = envior;
		}
		for(int a = 0; a < width; a++){
					for(int b = 0; b < length; b++){
						if((map[a][b][level] == type) && !(a == location[0] && b == location[1])){
							if((a - location[0]) != 0){
								dxdy[0] += width / (a - location[0]);
							}
							if((b - location[1]) != 0){
								dxdy[1] += length / (b - location[1]);
							}
						}
					}
				}

		return dxdy;

}

int Swarmbot::getMap(int x, int y, int level){
	return map[x][y][level];
}

