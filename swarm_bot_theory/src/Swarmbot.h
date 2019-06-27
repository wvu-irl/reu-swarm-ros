/*
 * Swarmbot.h
 *
 *  Created on: May 29, 2019
 *      Author: Trevs
 */
//test
#ifndef SWARMBOT_H_
#define SWARMBOT_H_

class Swarmbot {
private:
	int foodLevel;
	int habitatLevel;
	int old;
	int coord[2];

public:

	Swarmbot(){};
	int location[2];
	Swarmbot(int xy[2], int m[10][10][2]);
	virtual ~Swarmbot();
	int* findFood();
	int* findHabitat();
	void move();
	void age();
	void printCoord();
	void die();
	void updateSMap(int m[10][10][2]);

	const int* getCoord() const {
		return coord;
	}

	int getFoodLevel() const {
		return foodLevel;
	}

	void setFoodLevel(int foodLevel) {
		this->foodLevel = foodLevel;
	}

	int getHabitatLevel() const {
		return habitatLevel;
	}

	void setHabitatLevel(int habitatLevel) {
		this->habitatLevel = habitatLevel;
	}

	const int* getLocation() const {
		return location;
	}

	int getOld() const {
		return old;
	}

	void setOld(int old) {
		this->old = old;
	}


};

#endif /* SWARMBOT_H_ */
