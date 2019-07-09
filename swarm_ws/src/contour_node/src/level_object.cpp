#include <contour_node/level_object.h>

levelObject::levelObject(void)
{
    origin.first = 0.0;
    origin.second = 0.0;
    name = "";
    selected = false;
}

levelObject::levelObject(std::pair<double, double> _orig, std::string _name)
    : origin(_orig), name(_name), selected(false) {
}

levelObject::levelObject(double _xorg, double _yorg, std::string _name)
{
    origin = std::pair<double, double>(_xorg, _yorg);
    name = _name;
}

levelObject::~levelObject(void) {}

std::pair<double, double> levelObject::getOrigin(void) {return origin;}

void levelObject::setName(std::string _s) {name = _s;}
std::string levelObject::getName() {return name;}

void levelObject::select(void) {selected = true;}
void levelObject::deselect(void) {selected = false;}
bool levelObject::isSelected(void) {return selected;}