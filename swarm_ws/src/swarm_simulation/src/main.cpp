#include <iostream>
#include <swarm_simulation/Sim.h>
#include <ros/ros.h>
#include<cmath.h>
#include<stdlib.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>

int main(int argc, char **argv)
{
    Sim sim;
  	ros::init(argc, argv, "simulation");
  	ros::NodeHandle n;

  	/*wvu_std_msgs::robot_command a;
  	a.rid = 'NY';
		cmd.r = 2;
		cmd.theta = M_PI/4;

  	wvu_std_msgs::robot_command b;
  	cmd.rid = 'NH';
		cmd.r = 2;
		cmd.theta = -M_PI/2;

  	wvu_std_msgs::robot_command c;
  	cmd.rid = 'WV';
		cmd.r = 2;
		cmd.theta = M_PI;

  	std::vector<wvu_std_msgs::robot_command> test_array{a,b,c,d};*/


  	//while(ros::ok())
//  	{ros::Subscriber sub = n.subscribe("final_execute", 1000, botCallback); //Subscribes to the Vicon
//  	wvu_std__msgs::robot_command_array trc_vector = *(ros::topic::waitForMessage < wvu_swarm_std_msgs::robot_command_array
//				> ("final_execute"));}

    sim.Run();
    return 0;
}

//Aleks Hatfield
//May 29, 2019
//WVU Robotics REU


