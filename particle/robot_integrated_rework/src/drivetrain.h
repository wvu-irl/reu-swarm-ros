#ifndef DRIVETRAIN_H
#define DRIVETRAIN_H

#include <Particle.h>
#define DIFF_DRIVE_DEBUG 1

class Drivetrain
{
public:
    Drivetrain(void) {}
    virtual void init(void) = 0;

    /*
     * Commands the servos based on the desired direction of travel
     */
    virtual void drive(double _theta, double _speed, float yaw) = 0;

protected:
    // TODO: damper/max speed variables?
};

class DiffDrive : public Drivetrain
{
public:
    DiffDrive(void);
    void init(void);
    void drive(double _theta, double _speed, float yaw);

private:
    Servo servLeft, servRight;
    float lv, lw; //defines the nature of the turn, the constants for linear and angular velocity.

    float error;
    float oldError;
    float dError;
    float omega;
    float kp;
    float kd;
    float aCommand;
    float bCommand;
    double tStep;
    unsigned long tstep1;
    bool start;
    float l;
};

class Holonomic : public Drivetrain
{
public:
    // TODO
private:
    // TODO
    
};

#endif