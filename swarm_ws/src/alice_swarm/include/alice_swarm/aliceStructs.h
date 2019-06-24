#ifndef ALICESTRUCTS_H
#define ALICESTRUCTS_H

#include <string>
#include <vector>
#include <list>
namespace AliceStructs {

	typedef struct {
		float dir;
		float spd;
		float dis;
		float pri;
		int name;
	} ideal; //stores a vector, along with how far away the vector comes from and from who

	typedef struct {
		float dir;
		float mag;
	} vel; //simplest form of a vector

	typedef struct {
		float dir;
		float dis;
		float ang;
		int name;
		int sid;
	} neighbor; //stores the location of a neighbor, along with its name and swarm id

	typedef struct {
		float dir;
		float dis;
	}  obj; //simplest form of a point


	typedef struct {
		std::vector<neighbor> neighbors;
		std::vector<obj> obstacles;
		std::vector<obj> targets;
		std::vector<ideal> flows;
		int name;
		int sid;
	} mail; //stores all info that a singular robot is given
};

#endif
