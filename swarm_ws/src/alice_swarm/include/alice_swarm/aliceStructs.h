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
#include <wvu_swarm_std_msgs/chargers.h>
#include <swarm_server/battery_states.h>

namespace AliceStructs
{

struct vector_2f
{
	double x, y;
	double dx, dy;
	bool valid;
};

/*
 * A vector for Alice to follow
 */
typedef struct
{
	float x;
	float spd;
	float y;
	float pri;
	ros::Time time;
	std::vector<int> observer_names;
} flow;

/*
 * A velocity
 */
typedef struct
{
	float dir;
	float mag;
} vel;

/*
 * Another robot near Alice
 */
typedef struct
{
	float x;
	float y;
	float tar_x;
	float tar_y;
	float ang;
	int name;
	int sid;
} neighbor;

/*
 * An object, usually an obstacle to be avoided
 */
typedef struct
{
	float x_rad;
	float y_rad;
	float x_off;
	float y_off;
	float theta_offset;
	ros::Time time;
	std::vector<int> observer_names;
} obj;

/*
 * A point
 */
typedef struct
{
	float x;
	float y;
	float z;
	ros::Time time;
	std::vector<int> observer_names;
} pnt;

/*
 * A pose
 */
typedef struct
{
	float x;
	float y;
	float z;
	float heading;
	ros::Time time;
	std::vector<int> observer_names;
} pose;


typedef struct
{
	float x;
	float y;
	bool occupied;
} charger;

/*
 * A box of all these structs, for easy transport
 */
typedef struct
{
	std::vector<neighbor> neighbors;
	std::vector<obj> obstacles;
	std::vector<pnt> targets;
	std::vector<flow> flows;
	std::vector<wvu_swarm_std_msgs::charger> *abs_chargers; //pointer to absolute chargers
	std::vector<wvu_swarm_std_msgs::charger> rel_chargers; //copy of chargers with pos relative to each bot.
	std::vector<float> *priority; //{REST, CHARGE, CONTOUR, TARGET, EXPLORE}
	float xpos;
	float ypos;
	float contVal;
	float heading;
	int name;
	float vision;
	float battery_lvl;
	BATTERY_STATES battery_state;
	float energy;
	ros::Time time;
} mail;

};

#endif
