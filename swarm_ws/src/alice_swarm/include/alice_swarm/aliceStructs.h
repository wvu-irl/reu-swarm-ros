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
	} ideal;

	typedef struct {
		float dir;
		float mag;
	} vel;

	typedef struct {
		float dir;
		float dis;
		float ang;
		int name;
		int sid;
	} neighbor;

	typedef struct {
		float dir;
		float dis;
	}  obj;


	typedef struct {
		std::vector<neighbor> neighbors;
		std::vector<obj> obstacles;
		std::vector<obj> targets;
		std::vector<ideal> flows;
		int name;
		int sid;
	} mail;
};

#endif