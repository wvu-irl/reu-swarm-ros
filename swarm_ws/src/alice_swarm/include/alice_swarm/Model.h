/*
 * Model
 *
 * This class contains all the information a given robot knows about its environment.
 * It has a rule set, which it uses to generate an ideal vector, which is simply what a robot would do if
 * if controlled the swarm alone.
 *
 * Authors: Jeongwoo Seo and Casey Edmonds-Estes
 */

#ifndef MODEL_H_
#define MODEL_H_

#include "alice_swarm/aliceStructs.h"
#include "alice_swarm/Rules.h"

class Model
{
public:

	/*
	 * Information Alice knows about her environment
	 */
	std::list <AliceStructs::obj> obstacles;
	std::list <AliceStructs::neighbor> robots;
	std::list <AliceStructs::obj> targets;
	std::list <AliceStructs::ideal> flows;

	Rules rules; //See Rules for more info

	/*
	 * Dummy constructor for the compiler
	 */
	Model();

	Model(int _name,int _sid);

	/*
	 * Generates an ideal vector from a given set of rules
	 * To adjust the rules, simply comment and uncomment them
	 * The second parameter each rules gets passed is it's strength, or to what extent the robot will priorotize it
	 * For more info, see rules.cpp
	 */
	AliceStructs::ideal generateIdeal();

	/*
	 * Adds information to the model
	 */
	void addToModel(AliceStructs::mail toAdd);

	/*
	 * Clears the model to prevent memory issues
	 */
	void clear();

private:

	/*
	 * Information Alice knows about herself
	 */
	int name;
	int sid;
};

#endif /* MODEL_H_ */
