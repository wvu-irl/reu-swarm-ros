#include <contour_node/level_description.h>
#include <math.h>

double map::calculate(contour_node::map_level ml,
		wvu_swarm_std_msgs::vicon_point loc)
{
	double x = loc.x;
	double y = loc.y;

	double z = 0;
	for (size_t i = 0; i < ml.functions.size(); i++)
	{
		contour_node::gaussian curr_eq = ml.functions[i];
		double theta = curr_eq.ellipse.theta_offset
				+ (x == 0 ?
						(y > 0 ? M_PI_2 : 3 * M_PI_2) : (atan(y / x) + (x < 0 ? M_PI : 0)));
		double r = sqrt(x * x + y * y);

		double a = curr_eq.ellipse.x_rad;
		double b = curr_eq.ellipse.y_rad;

		double x_app = r * cos(theta);
		double y_app = r * sin(theta);

		double re = sqrt(a * a * x_app * x_app + y_app * y_app * b * b) / (a * b);
		z += curr_eq.amplitude * pow(M_E, (-x*x) / 2.0);
	}

	return z;
}
