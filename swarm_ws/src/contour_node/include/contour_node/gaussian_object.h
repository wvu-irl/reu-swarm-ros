#ifndef GAUSSIANOBJECT_H
#define GAUSSIANOBJECT_H

#include "ellipse_object.h"
#include "level_description.h"
#include <wvu_swarm_std_msgs/obstacle.h>

class gaussianObject: public ellipseObject
{
public:
	gaussianObject(void);
	gaussianObject(std::pair<double, double> _orig, std::string _name,
			std::pair<double, double> radii, double _theta, double _ampl,
			levelType _lvl);
	gaussianObject(double _xorg, double _yorg, std::string _name, double _xrad,
			double _yrad, double _theta, double _ampl, levelType _lvl);

	gaussianObject(wvu_swarm_std_msgs::obstacle message);
	virtual ~gaussianObject(void);

	void nuiManipulate(double _x, double _y, double _z);

	double getAmplitude(void);

	wvu_swarm_std_msgs::obstacle getGaussianMessage(void);

protected:
	double amplitude;
};

#endif /* GAUSSIANOBJECT_H */

