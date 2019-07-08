#ifndef ELLIPSEOBJECT_H
#define ELLIPSEOBJECT_H

#include "level_object.h"

class ellipseObject : public levelObject {
public:
    ellipseObject(void);
    ellipseObject(std::pair<double, double> _orig, std::string _name,
        std::pair<double, double> _radii, double _theta);
    ellipseObject(double _xorg, double _yorg, std::string _name,
        double _xrad, double _yrad, double _theta);
    virtual ~ellipseObject(void);
    
    void nuiManipulate(double _x, double _y, double _z);
    
    std::pair<double, double> getRadii(void);
    double getTheta(void);
    
private:
    std::pair<double, double> radii; // x, y
    double theta;
};

#endif /* ELLIPSEOBJECT_H */

