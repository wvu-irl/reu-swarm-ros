#include <contour_node/gaussian_object.h>

gaussianObject::gaussianObject(void) {
}

gaussianObject::gaussianObject(std::pair<double, double> _orig, std::string _name,
    std::pair<double, double> _radii, double _theta, double _ampl, levelType _lvl)
    : ellipseObject(_orig, _name, _radii, _theta), amplitude(_ampl), level(_lvl) {
}

gaussianObject::gaussianObject(double _xorg, double _yorg, std::string _name,
    double _xrad, double _yrad, double _theta, double _ampl, levelType _lvl)
    : ellipseObject(_xorg, _yorg, _name, _xrad, _yrad, _theta), amplitude(_ampl), level(_lvl) {
}

gaussianObject::~gaussianObject(void) {}

void gaussianObject::nuiManipulate(double _x, double _y, double _z)
{
    // If hand's height moved, change amplitude
    if(_z > 2.0) amplitude += 1.5;
    else if(_z < -2.0) amplitude -= 1.5;
    else if(abs(_z) > 0.5) amplitude += (_z - 0.5);
    
    // If hand's location moved, move the origin
    if(_x > 2.0) origin.first += 1.5;
    else if(_x < -2.0) origin.second -= 1.5;
    else if(abs(_x) > 0.1) origin.first += (_x);
    if(_y > 2.0) origin.second += 1.5;
    else if(_y < -2.0) origin.second -= 1.5;
    else if(abs(_y) > 0.1) origin.second += (_y);
}

double gaussianObject::getAmplitude(void) {return amplitude;}

levelType gaussianObject::getLevel(void) {return level;}
void gaussianObject::setLevel(levelType _lvl) {level = _lvl;}