#include <iostream>
#include <swarm_simulation/Sim.h>
#include <ros/ros.h>

int main(int argc, char **argv)
{
    Hawk_Sim sim;
  	sim.run();
    return 0;
}
