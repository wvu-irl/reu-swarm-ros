#include "ros/ros.h"
#include "std_msgs/String.h"
#include "stdlib.h"
#include <sstream>

#include <wvu_swarm_std_msgs/vicon_points.h>

void makeAndPublish(ros::Publisher _pub)
{
  wvu_swarm_std_msgs::vicon_points vp_vector;
  int n = 4; //number of obs points
  std::pair<float, float> o1(100, 100);
	std::pair<float, float> o2(200, 200);
	std::pair<float, float> o3(-100, -100);
	std::pair<float, float> o4(-200, -200);
	std::pair<float, float> pair_array[n] = {o1, o2, o3, o4 };
  for (int i = 0; i < n; i ++)
  {
    wvu_swarm_std_msgs::vicon_point cur;
    cur.x = pair_array[i].first;
    cur.y = pair_array[i].second;
    vp_vector.point.push_back(cur);
  }
  _pub.publish(vp_vector);

}
int main(int argc, char **argv)
{
  ros::init(argc, argv, "virtual_obstacle");
  ros::NodeHandle n;
  ros::Publisher pub = n.advertise<wvu_swarm_std_msgs::vicon_points>("virtual_obstacles",1000);

  while (ros::ok())
  {
		makeAndPublish(pub);
		ros::spinOnce();
  }
}
