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
#include <ros/ros.h>
namespace AliceStructs {

/*
 * A vector for Alice to follow
 */
	typedef struct {
		float x;
		float spd;
		float y;
		float pri;
		ros::Time time;
		int observer_name;
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
		float x;
		float y;
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
		float x_off;
		float y_off;
		float theta_offset;
		ros::Time time;
		int observer_name;
	} obj;

	/*
	 * A point
	 */
	typedef struct {
		float x;
		float y;
		float z;
		ros::Time time;
		int observer_name;
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
		float xpos;
		float ypos;
		float level;
		ros::Time time;
		float vision;
	} mail;
};

#endif
