
#include "ros/ros.h"
#include "wvu_swarm_std_msgs/alice_mail.h"
#include "alice_swarm/Hub.h"
#include "alice_swarm/Alice.h"
#include "alice_swarm/aliceStructs.h"
#include <sstream>
#include <map>
#include <chrono>

int main(int argc, char **argv)
{

	wvu_swarm_std_msgs::alice_mail mail;
	wvu_swarm_std_msgs::point_mail tar;
	tar.x = 20;
	tar.y = 0;
	wvu_swarm_std_msgs::ellipse obj;
	obj.x_rad = 3;
	obj.y_rad = 5;
	obj.theta_offset = 0;
	obj.offset_x = 10;
	obj.offset_y = 0;
	mail.obsMail.push_back(obj);
	wvu_swarm_std_msgs::ellipse obj2;
	obj2.x_rad = 5;
	obj2.y_rad = 5;
	obj2.theta_offset = 0;
	obj2.offset_x = 10;
	obj2.offset_y = 3;
	mail.obsMail.push_back(obj2);
	mail.targetMail.push_back(tar);
	Alice alice;
	alice = Alice(mail);
	alice.generateVel();
	return 0;
}
