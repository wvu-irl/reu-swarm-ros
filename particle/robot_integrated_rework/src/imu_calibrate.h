#ifndef IMUCALIBRATE_H
#define IMUCALIBRATE_H

#include <Particle.h>
#include <MPU6050.h>

class IMUCalibrate
{
public:
    IMUCalibrate(void);
    void init(void);
    
    /*
     *  returns the desired relative heading after a timestep using the imu.
     */
    float getIMUHeading(float _otheta);

private:
    MPU6050 accelGyro;
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    // Pitch, Roll and Yaw values (seemed to be unused)
    // float pitch = 0;
    // float roll = 0;
     float yaw;
     float oldYaw;

    // Time values
    float timeStep;
    int t1;


};

#endif