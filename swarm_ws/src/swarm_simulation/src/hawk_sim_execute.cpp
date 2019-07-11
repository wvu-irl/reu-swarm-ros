#include <iostream>
#include <swarm_simulation/Sim.h>
#include <ros/ros.h>
#include <swarm_simulation/hawk_sim.h>

int main(int argc, char **argv)
{
    Hawk_Sim sim;
		ros::init(argc, argv, "hawk_simulation");
		ros::NodeHandle n;
		sim.run(n);
    return 0;
}
