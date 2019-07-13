#include "imu_calibrate.h"
#include "math.h"
#define IMU_DEBUG 1

IMUCalibrate::IMUCalibrate(void)
{
    yaw = 0;
    oldYaw = 0;
    timeStep = 0;
    t1 = 0;
}

void IMUCalibrate::init(void)
{
    accelGyro.initialize();
    #if IMU_DEBUG
    Serial.println("IMU: Testing device connections...");
    Serial.println(accelGyro.testConnection() ? "IMU: MPU6050 connection successful" : "IMU: MPU6050 connection failed");
    #endif
}

float IMUCalibrate::getIMUHeading(float _otheta)
{
    int t2 = millis();
    timeStep = t2 - t1;
    oldYaw = gz / 131;
    accelGyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    yaw = gz / 131;

    float theta = _otheta - ((yaw + oldYaw) / 2) * timeStep * .001;
    t1 = t2;
    return theta;
}

#endif 