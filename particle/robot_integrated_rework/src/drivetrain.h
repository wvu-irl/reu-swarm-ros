#ifndef DRIVETRAIN_H
#define DRIVETRAIN_H

#include <Particle.h>

class Drivetrain {
public:
    Drivetrain(void) {}
    virtual void init(void) = 0;
    virtual void drive(double heading, double speed) = 0;
protected:
    // TODO: damper/max speed variables?
};

class DiffDrive : public Drivetrain {
public:
    DiffDrive(void);
    void init(void);
private:
    Servo servLeft, servRight;
};

class Holonomic : public Drivetrain {
public:
    // TODO
private:
    // TODO
};

#endif