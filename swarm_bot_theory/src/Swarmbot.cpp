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

	//moving constants
	int const up = 0;
	int const down = 1;
	int const left = 2;
	int const right = 3;


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

	//known world Map,x, y,  [food, habitat, robots, human, unknown]
	double ***KnownMap;

	//map you can see
	double ***LocalMap;

	//add in MDP
	int const Actions[][] = {{0,1}, {1, 1}, {0, 1}, {-1, 1}, {-1,0},{-1,-1},{0,-1}, {1,-1}};

	//robot is dead = true, robot is alive = false
	bool isDead;

	int direction;


//-----------------------------------------------------------------constructors
Swarmbot::Swarmbot(int w, int l, int x, int y, int personal[8]) {
	direction = 0;
	lifetime = 0;
for(int g = 0; g < 8; g++){
	personality[g] = personal[g];
}
	health[0]= 100;
	health[1]= 100;
	health[2]= 100;
	health[3]= 100;
	health[4]= 100;
	isDead = false;
	location[0] = x;
	location[1] = y;
	widthWorld = w;
	lengthWorld = l;
	//known map holds the values for the MDP
	KnownMap = new double**[widthWorld];
	//Local map holds the Rewards within the sight range
	LocalMap = new double**[widthWorld];
	for(int x = 0; x < widthWorld; x++){
		KnownMap[x] = new double*[lengthWorld];
		for(int y = 0; y < lengthWorld; y++){
			KnownMap[x][y] = new double[5];
		}
	}

	for(int x = 0; x < widthWorld; x++){
		for(int y = 0; y < lengthWorld; y++){
			KnownMap[x][y][foods] = empty;
			KnownMap[x][y][habitats] = empty;
			KnownMap[x][y][robots]= empty;
			KnownMap[x][y][humans] = empty;
			KnownMap[x][y][unknowns] = 1;
		}
	}
	for(int x = 0; x < widthWorld; x++){
		LocalMap[x] = new double*[lengthWorld];
		for(int y = 0; y < lengthWorld; y++){
			LocalMap[x][y] = new double[5];
		}
	}
	for(int x = 0; x < sightRng; x++){
		for(int y = 0; y < sightRng; y++){
			KnownMap[x][y][foods] = empty;
			KnownMap[x][y][habitats] = empty;
			KnownMap[x][y][robots]= empty;
			KnownMap[x][y][humans] = empty;
			KnownMap[x][y][unknowns] = 1;
		}
	}


}


Swarmbot::~Swarmbot() {
}

//look around
void Swarmbot::lookAround(double localWorld[sightRng][sightRng][5]){
	for(int x = 0; x< sightRng; x++)
		for(int y = 0; y < sightRng; y++)
			for(int i = 0; i < 5; i++)
				LocalMap[x][y][i] = localWorld[x][y][i];
}

//------------------------------------------------------finds
/*
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

*/

double* ResourceVector;
double* Swarmbot::resultantResourceVector(int resource){

	double dx = 0;
	double dy = 0;
	double mag = 0;

	for(int x = 0; x < widthWorld; x++){
		for (int y = 0; y < lengthWorld; y++){
			double value = KnownMap[x][y][resource];
			double dist =  sqrt(x^2 + y^2);
			if(dist != 0){
				dx += value* x/dist;
				dy += value * y/dist;
			}
		}
	}
	ResourceVector = new double[dx, dy];
	return ResourceVector;

}

//----------------------------------------------------actions
void Swarmbot::move() {

	//[communication level, food level, habitat level, human influence level, exploration level]a
	//determine local MDP for each resource
	double localValues = new double[2*sightRng][2*sightRng][5];
	//for each resource
	for(int r = 0; r < 5; r++)
		//for each state
		for(int x = location[0] - sightRng; x < location[0]+sightRng; x++){
			for(int y = location[1] - sightRng; y < location[1]+sightRng; y++){
				//for each action
				for(int a = 0; a < 9; a++){
					//remember highest value
					double locationActionValue= LocalMap[x-location[0]+Actions[a][0]][y - location[1]+Actions[a][1]][r] + personality[gamma]* KnownMap[x+Actions[a][0]][y+Actions[a][1]][r];
					if(locationActionValue > localValues[x][y][r])
						localValues[x][y][r] = locationActionValue;
				}
				//update the known map values
				KnownMap[x][y][r] = localValues[x][y][r];
			}
		}

	//multiply local values by personality and health level then sum together and store highest value coords
	double resultantLocalValues = new double[2*sightRng][2*sightRng];

	for (int r = 0; r < 5; r++){
		for(int x = 0; x < 2*sightRng; x++){
			for(int y = 0; y < 2*sightRng; y++){
				resultantLocalValues[x][y] += KnownMap[x][y][r]*personality[r]*(100 - health[r]);

			}
		}
	}
	int dx = 0;
	int dy = 0;
	int maxValue = 0;
		for(int x = 0; x < 2*sightRng; x++){
			for(int y = 0; y< 2*sightRng; y++){
				if(resultantLocalValues[x][y] > maxValue){
					dx = x;
					dy = y;
					maxValue = resultantLocalValues[x][y];
				}
			}
		}
	location[0] = location[0] + dx - sightRng;
	location[1] = location[1] + dy - sightRng;


}

void Swarmbot::getResource(int Resource){
	bool foundResource = false;
	while(!foundResource)
		for(int x = location[0] - sightRng; x < location[0]+sightRng; x++){
			for(int y = location[1] - sightRng; y < location[1]+sightRng; y++){

			}
		}
}

void Swarmbot::aging() {

	//get resources
	//int resource = map[location[0]][location[1]][envior];
	/*int onHuman = map[location[0]][location[1]][humans];

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
	}*/


}

void Swarmbot::die() {
	/*int d = 0;
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
*/
}


//--------------------------------------------------------
void Swarmbot::updateMap(int x, int y, int level, int value) {
	/*map[x][y][level] = value;

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

	return avgLoc;*/

}
int dxdy[2] = {0,0};
/*int* Swarmbot::furtherLessdxdy(int type){

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

}*/

int Swarmbot::getMap(int x, int y, int level){
	return 5; //map[x][y][level];
}

