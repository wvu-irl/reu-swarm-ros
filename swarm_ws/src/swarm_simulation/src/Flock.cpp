#include <swarm_simulation/Body.h>
#include <swarm_simulation/Flock.h>

//Flock Functions from Flock.h
//-----------------------------

int Flock::getSize()
{
    return flock.size();
}

Body Flock::getBody(int i)
{
    return flock[i];
}

void Flock::addBody(Body b)
{
    flock.push_back(b);
}

// Runs the run function for every body in the flock checking against the flock
// itself. Which in turn applies all the rules to the flock.
void Flock::flocking()
{
    for (int i = 0; i < flock.size(); i++)
        flock[i].run(flock);
}
