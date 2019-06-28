/*
 * Stores all the data types for Alice
 *
 * Author: Casey Edmonds-Estes
 */
#ifndef ALICESTRUCTS_H
#define ALICESTRUCTS_H

#include <string>
#include <vector>
#include <list>
namespace AliceStructs {

/*
 * A vector for Alice to follow
 */
	typedef struct {
		float dir;
		float spd;
		float dis;
		float pri;
		int t;
	} flow;

	/*
	 * A velocity
	 */
	typedef struct {
		float dir;
		float mag;
	} vel;

	/*
	 * Another robot near Alice
	 */
	typedef struct {
		float dir;
		float dis;
		float ang;
		int name;
		int sid;
	} neighbor;

	/*
	 * An object, usually an obstacle to be avoided
	 */
	typedef struct {
		float x_rad;
		float y_rad;
		float theta_offset;
		int t;
	} obj;

	/*
	 * A point
	 */
	typedef struct {
		float x;
		float y;
		float z;
		int t;
	} pnt;

	/*
	 * A box of all these structs, for easy transport
	 */
	typedef struct {
		std::vector<neighbor> neighbors;
		std::vector<obj> obstacles;
		std::vector<pnt> targets;
		std::vector<flow> flows;
		int name;
	} mail;
};

#endif
