#include <contour_node/level_description.h>
#include <math.h>

double map_ns::calculate(wvu_swarm_std_msgs::map_level ml,
		wvu_swarm_std_msgs::vicon_point loc)
{
	double x = loc.x;
	double y = loc.y;

	double z = 0;
	for (size_t i = 0; i < ml.functions.size(); i++)
	{
		wvu_swarm_std_msgs::gaussian curr_eq = ml.functions[i];
		double theta = curr_eq.ellipse.theta_offset
				+ (x == 0 ?
						(y > 0 ? M_PI_2 : 3 * M_PI_2) : (atan(y / x) + (y < 0 ? M_PI : 0)));
		double r = sqrt(x * x + y * y);

		double a = curr_eq.ellipse.x_rad;
		double b = curr_eq.ellipse.y_rad;

		double x_app = r * cos(theta);
		double y_app = r * sin(theta);

		double re = sqrt(a * a * x_app * x_app + y_app * y_app * b * b) / (a * b);
		z += curr_eq.amplitude * pow(M_E, (-re * re) / 2.0);
	}

	return z;
}

wvu_swarm_std_msgs::map_level map_ns::combineLevels(
		wvu_swarm_std_msgs::map_level a, wvu_swarm_std_msgs::map_level b)
{
	wvu_swarm_std_msgs::map_level n_lev;
	for (size_t i = 0; i < a.functions.size(); i++)
		n_lev.functions.push_back(a.functions.at(i));

	for (size_t i = 0; i < b.functions.size(); i++)
		n_lev.functions.push_back(b.functions.at(i));

	return n_lev;
}
