#include <contour_node/ellipse_object.h>

ellipseObject::ellipseObject(void)
{
}

ellipseObject::ellipseObject(std::pair<double, double> _orig, std::string _name,
		std::pair<double, double> _radii, double _theta, levelType _lvl) :
		levelObject(_orig, _name, _lvl), radii(_radii), theta(_theta)
{
}

ellipseObject::ellipseObject(double _xorg, double _yorg, std::string _name,
		double _xrad, double _yrad, double _theta, levelType _lvl) :
		levelObject(_xorg, _yorg, _name, _lvl)
{
	radii = std::pair<double, double>(_xrad, _yrad);
	theta = _theta;
}

ellipseObject::ellipseObject(wvu_swarm_std_msgs::ellipse _msg,
		std::string _name, levelType _lvl) :
		levelObject(_msg.offset_x, _msg.offset_y, _name, _lvl), theta(
				_msg.theta_offset), radii(
				std::pair<double, double>(_msg.x_rad, _msg.y_rad))
{
}

ellipseObject::~ellipseObject(void)
{
}

void ellipseObject::nuiManipulate(double _x, double _y, double _z)
{
	// TODO: change the ellipse as the hand moves
}

std::pair<double, double> ellipseObject::getRadii(void)
{
	return radii;
}
double ellipseObject::getTheta(void)
{
	return theta;
}

wvu_swarm_std_msgs::ellipse ellipseObject::getEllipseMessage(void)
{
	wvu_swarm_std_msgs::ellipse el;
	el.x_rad = radii.first;
	el.y_rad = radii.second;
	el.offset_x = origin.first;
	el.offset_y = origin.second;
	el.theta_offset = theta;
	return el;
}
