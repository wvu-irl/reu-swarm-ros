#include <iostream>
#include <swarm_simulation/Sim.h>
#include <ros/ros.h>

int main(int argc, char **argv)
{
    Hawk_Sim sim;
		ros::init(argc, argv, "Hawk_Simulation");
		ros::NodeHandle n;
  	sim.run(n);
    return 0;
}
