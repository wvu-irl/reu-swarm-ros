#include <iostream>
#include <swarm_simulation/Sim.h>
#include <ros/ros.h>

int main(int argc, char **argv)
{
    Sim sim;
  	ros::init(argc, argv, "simulation");
  	ros::NodeHandle n;
    sim.Run();
    return 0;
}

//Aleks Hatfield
//May 29, 2019
//WVU Robotics REU

//Priority:
/*
 *Sim objects
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */
