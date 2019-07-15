#include "drivetrain.h"
#include "math.h"

DiffDrive::DiffDrive(void)
{
    servRight.attach(LEFTPIN);
    servLeft.attach(RIGHTPIN);
    lv = 3;
    lw = 2;
}

void DiffDrive::init(void)
{
    //no-op for now?
}

void DiffDrive::drive(double _theta, double _speed)
{
    float v = lv * cos(_theta * 3.14 / 180);
    float w = lw * sin(_theta * 3.14 / 180);

    if ((_theta >= 0 && _theta <= 90) || _theta > 270)
    {
        servLeft.write(90 + 10 * (v - w));
        servRight.write(90 - 10 * (v + w));
    }
    else if (_theta > 90 && _theta <= 180)
    { //we still want this because we want linear velocity to be non-negative ( change for pd)
        servLeft.write(90 - 10 * lw);
        servRight.write(90 - 10 * lw);
    }
    else if (_theta > 180 && _theta < 270)
    {
        servLeft.write(90 + 10 * lw);
        servRight.write(90 + 10 * lw);
    }
    else //if somehow the command is invalid
    {
        servLeft.write(90);
        servRight.write(90);
    }
}

// Completely disconnects servos to stop
void DiffDrive::fullStop(void)
{
    servLeft.detach();
    servRight.detach();
}

// Reattaches servos. Call this after a fullStop
void DiffDrive::restart(void)
{
    servRight.attach(LEFTPIN);
    servLeft.attach(RIGHTPIN);
}