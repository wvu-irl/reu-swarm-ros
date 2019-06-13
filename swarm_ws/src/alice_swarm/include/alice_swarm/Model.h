#ifndef MODEL_H_
#define MODEL_H_

#include "aliceStructs.h"
#include "Rules.h"

class Model
{
public:
	std::list <obj> obstacles;
	std::list <neighbor> robots;
	std::list <tar> targets;

	Model();

	ideal generateIdeal();

	void addToModel(mail toAdd);

	void clear();

private:
	rules Rules;


};

#endif /* MODEL_H_ */
