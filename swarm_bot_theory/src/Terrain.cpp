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
#include <cstdlib>
#include <list>
#include <iostream>

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


Terrain::Terrain(int w, int h, int r) {
	twidth = w;
	tlength = h;
	numbots = r;
	world = new int**[twidth];
		for(int x = 0; x < twidth; x++){
			world[x] = new int*[tlength];
			for(int y = 0; y < tlength; y++){
				world[x][y] = new int[3];
			}
		}

		for(int x = 0; x < twidth; x++){
			for(int y = 0; y < tlength; y++){
				int r = rand() % 3;
				world[x][y][envior] = r;
				world[x][y][robots]= 0;
				world[x][y][humans] = 0;
			}
		}

		//world[0][5][envior] = food;
		//world[5][0][envior] = habitat;
		world[5][5][humans] = human;
		world[4][5][humans] = human;
		world[6][5][humans] = human;
		world[5][4][humans] = human;
		world[5][6][humans] = human;


		for(int n = 0; n < r; n++){
				bool broken = false;
				for(int x = 0; x < twidth; x++){
					for(int y = 0; y < tlength; y++){
						if(!isRobotThere(x,y)){
							world[x][y][robots] = robot;
							bots.emplace_back(Swarmbot(twidth, tlength, x, y));
							bots.at(n).updateMap(x, y, envior,world[x][y][envior]);
							bots.at(n).updateMap(x, y, robots,world[x][y][robots]);
							bots.at(n).updateMap(x,y, humans, world[x][y][humans]);
							broken = true;
							break;
						}
					}
					if(broken){

						break;
					}
				}

		}


}

Terrain::~Terrain() {
}


void Terrain::addRobot() {
//	Swarmbot bot = Swarmbot(twidth, tlength);
//	bots.emplace_back(bot);
//	bool broken = false;
//	for(int x = 0; x < twidth; x++){
//		for(int y = 0; y < tlength; y++){
//			if(!isRobotThere(x,y)){
//				world[x][y][robots] = robot;
//				bot.setLocation(x,y);
//				broken = true;
//				break;
//			}
//		}
//		if(broken){
//			break;
//		}
//	}
}

int Terrain::getWorld(int x, int y, int level) {
	return world[x][y][level];
}

bool Terrain::isRobotThere(int x, int y) {
	if(world[x][y][robots] == robot){
		return true;
	}
	else{
		return false;
	}
}

void Terrain::setHumanInfluence(std::list<int*> coords) {
	for(int x = 0; x < twidth; x++){
			for(int y = 0; y < tlength; y++){
				world[x][y][humans] = empty;
			}
		}
	for(int* coord : coords){
		world[*coord][*(coord+1)][humans] = human;
	}
}


void Terrain::printMap(int level){

	for (int y = 0; y < tlength; y++)
			{
				for (int x = 0; x < twidth; x++)
				{
					//cout << 4;
					std::cout <<world[x][y][level]<< " ";

				}
				std::cout << std::endl;
			}
			std::cout << "---------------------------------------------------------------------------------" << std::endl;

}



void Terrain::updateWorld() {
	for(int x = 0; x < twidth; x++){
		for(int y = 0; y < tlength; y++){
			world[x][y][robots] = empty;
		}
	}


	//full communication-------------------------------------------------------------
	for(int i = 0; i < numbots; i++){

		for(int x = 0; x < twidth; x++){
			for(int y =0; y < tlength; y++){
				bots.at(i).updateMap(x,y,robots,empty);
				bots.at(i).updateMap(x,y,humans, empty);
			}
		}
		for(int j = 0; j < numbots; j++){
			for(int a = 0; a < twidth; a++){
				for(int b = 0; b <tlength; b++){
					bots.at(i).updateMap(a,b,envior,bots.at(j).getMap(a,b,envior));
					bots.at(i).updateMap(a,b,humans,world[a][b][humans]);
					int* loc = bots.at(j).getLocation();
					bots.at(i).updateMap(*(loc), *(loc+1), robots, robot);

					bots.at(i).Values[a][b] = std::max(bots.at(i).Values[a][b], bots.at(j).Values[a][b]);
				}
			}

		}
		//--------------------------------------------------------------------------
		int* j = bots.at(i).getHealth();
		int health1[5] = {*j, *(j+1), *(j+2), *(j+3), *(j+4)};
		std::cout<<"space: "<< health1[empty] << " food: " << health1[food]<< " habitat: "<<health1[habitat] << " human: "<<health1[human]<< " new "<< health1[unexplored] <<std::endl;
		bots.at(i).exist();



		int* coor = bots.at(i).getLocation();
		int x = *coor;
		int y = *(coor+1);
		world[x][y][robots] = robot;
		bots.at(i).updateMap(x, y, envior, world[x][y][envior]);
		bots.at(i).updateMap(x, y, robots, world[x][y][robots]);
		bots.at(i).updateMap(x, y, humans, world[x][y][humans]);

		//bots.at(i).knownWorld(envior);
		//bots.at(i).knownWorld(robots);
		//bots.at(i).knownWorld(humans);
		bots.at(i).printValues();

	}

}


void Terrain::testingFunction(){

	for(int x = 0; x < twidth; x++){
		for(int y = 0; y < tlength; y++){
			//int r = rand() % 3;
			world[x][y][envior] = 0;
			world[x][y][robots]= 0;
			world[x][y][humans] = 0;
		}
	}

		//food habitat space and human all work together
	//world[0][5][envior] = food;
	//world[5][0][envior] = habitat;
		world[0][5][humans] = human;
		world[4][0][humans] = human;


		for(int i = 0; i < numbots; i++){

			for(int x = 0; x< twidth; x++){
				for(int y = 0; y< tlength; y++){
					bots.at(i).updateMap(x, y, envior,world[x][y][envior]);
					bots.at(i).updateMap(x, y, robots,world[x][y][robots]);
					bots.at(i).updateMap(x, y, humans,world[x][y][humans]);
				}
			}
			bots.at(i).updateMap(0, 5, humans,human);
			bots.at(i).updateMap(4, 0, humans,human);

		}
}




