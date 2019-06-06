#include <iostream>
#include <vector>
#include "Body.h"

#ifndef FLOCK_H_
#define FLOCK_H_


// This file contains the class needed to create a flock of bodies. It utilizes
// the bodies class and initializes body flocks with parameters that can be
// specified. This class will be utilized in main.

class Flock {
public:
    vector<Body> flock;
    //Constructors
    Flock() {}
    // Accessor functions
    int getSize();
    Body getBody(int i);
    // Mutator Functions
    void addBody(Body b);
    void flocking();
};

#endif
