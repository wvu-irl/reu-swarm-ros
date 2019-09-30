//============================================================================
// Name        : swarm_bot_theory.cpp
// Author      : Trevor Smith
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "Terrain.h"
#include "Swarmbot.h"
#include <vector>
#include <list>

using namespace std;




int main() {

		int const empty = 0;
		int const food = 1;
		int const habitat = 2;
		int const human = 3;
		int const unexplored = 4;
		int const robot = 5;
		int const envior = 0;
		int const robots = 1;
		int const humans = 2;

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	int time = 0;
	Terrain ter = Terrain(10, 10, 3);

	//ter.testingFunction();

	ter.printMap(envior);
	ter.printMap(robots);
	ter.printMap(humans);




	for(int a = 0; a < 1000; a++){
		ter.updateWorld();
		time++;
		cout<<time<<endl;
	}




	return 0;
}
