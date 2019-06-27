//============================================================================
// Name        : swarm_bot_theory.cpp
// Author      : Trevor Smith
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
//test
#include <iostream>
#include "Terrain.h"
#include "Swarmbot.h"
#include <vector>

using namespace std;

void printmap(Terrain t){

	for (int y = 0; y < 10; y++)
			{
				for (int x = 0; x <10; x++)
				{
					//cout << 4;
					cout <<t.getPoint(x, y) << " ";

				}
				cout << endl;
			}
			cout << "---------------------------------------------------------------------------------" << endl;


}

void printRobotMap(Terrain ter){

	for (int y = 0; y < 10; y++)
		{
			for (int x = 0; x <10; x++)
				{
					cout<< ter.tMap[x][y][1]<< " ";
				}
			cout << endl;
		}
	cout<<"---------------------------------------------------------------------------------" <<endl;
}

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	int time;
	Terrain ter(10, 10);
	int xy[2];
	int* coord = ter.addRobot();
	xy[0] = *coord;
	xy[1]= *(coord+1);
	Swarmbot bot1( xy, ter.tMap);

	coord = ter.addRobot();
	xy[0] = *coord;
	xy[1]= *(coord+1);
	Swarmbot bot2(xy, ter.tMap);

	coord = ter.addRobot();
		xy[0] = *coord;
		xy[1]= *(coord+1);
		Swarmbot bot3(xy, ter.tMap);

		coord = ter.addRobot();
			xy[0] = *coord;
			xy[1]= *(coord+1);
			Swarmbot bot4(xy, ter.tMap);

			coord = ter.addRobot();
				xy[0] = *coord;
				xy[1]= *(coord+1);
				Swarmbot bot5(xy, ter.tMap);

	std::vector <Swarmbot> bots;
	bots.reserve(5);       //add more bots need more space
	bots.emplace_back(bot1);
	bots.emplace_back(bot2);
	bots.emplace_back(bot3);
	bots.emplace_back(bot4);
	bots.emplace_back(bot5);
	bot1.printCoord();
		printmap(ter);
		for (int t = 0; t < 100; t++) {

			std::vector <Swarmbot> temp;
			for(Swarmbot bot : bots){
				bot.move();
				bot.age();
				bot.die();
				cout << "food level = "<< bot.getFoodLevel() << "  habitat level = " << bot.getHabitatLevel() << endl;
				bot.printCoord();

				int x = *bot.getLocation();
				int y = *(bot.getLocation()+1);
				if(x != -1 || y != -1){
					temp.emplace_back(bot);
				}
				ter.updateTMap(bots);
				bot.updateSMap(ter.tMap);
			}
			bots.clear();
			for(Swarmbot bot : temp){
				bots.emplace_back(bot);
			}

			time ++;
			cout<<"time = " << time<< endl;

			if(bots.size() == 0){
				break;
			}
			printRobotMap(ter);

		}
		system("pause");



	return 0;
}
