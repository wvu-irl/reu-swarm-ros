/*
 * IMPORTANT:
 *  This file must use standard libraries only, such as cstdlib and Boost.
 *  Do NOT add any dependencies to ROS. This same header is used by the Nuitrack
 *  bridge UDP server, which cannot be compiled with ROS.
 *  Thanks :) nh
 */


#ifndef NUITRACK_DATA_H
#define NUITRACK_DATA_H

#include <string>

using std::string;

// Physical transform of the Kinect's lens against global frame
#define KINECT_TRAN_X 0.226 //-0.450
#define KINECT_TRAN_Y -0.966 //-0.107
#define KINECT_TRAN_Z 1.114 //0.585
#define KINECT_QUAT_X 0.54374 //0.543
#define KINECT_QUAT_Y 0.55073 //0.550
#define KINECT_QUAT_Z 0.44479 //0.444
#define KINECT_QUAT_W -0.45078 //0.450

enum class gestureType : char 
{
    NONE = '\0',
    WAVING = (char)1,
    SWIPE_LEFT = (char)2,
    SWIPE_RIGHT = (char)3,
    SWIPE_UP = (char)4,
    SWIPE_DOWN = (char)5,
    PUSH = (char)6
};

typedef struct xyz
{
    xyz() //Default Constructor
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }
    
    xyz(double _x, double _y, double _z) //Alternate Constructor
    {
        x = _x;
        y = _y;
        z = _z;
    }
    
    double x;
    double y;
    double z;
} xyz;

typedef struct nuiData
{
    nuiData() //Default Constructor
    {
        gestureFound = false;
        leftFound = false;
        rightFound = false;
        gestureData = gestureType::NONE;
        leftHand = xyz();
        leftWrist = xyz();
        rightHand = xyz();
        rightWrist = xyz();
        confLH = 0.0;
        confLW = 0.0;
        confRH = 0.0;
        confRW = 0.0;
        leftClick = false;
        rightClick = false;
    }

    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gT,
            xyz _lH, xyz _lW, xyz _rH, xyz _rW,
            double _cLH, double _cLW, double _cRH, double _cRW,
            bool _lC, bool _rC) //Alternate Constructor
    {
        gestureFound = _gF;
        leftFound = _lF;
        rightFound = _rF;
        gestureData = _gT;
        leftHand = _lH;
        leftWrist = _lW;
        rightHand = _rH;
        rightWrist = _rW;
        confLH = _cLH;
        confLW = _cLW;
        confRH = _cRH;
        confRW = _cRW;
        leftClick = _lC;
        rightClick = _rC;
    }
    
    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gT,
            double _lHX, double _lHY, double _lHZ,
            double _lWX, double _lWY, double _lWZ,
            double _rHX, double _rHY, double _rHZ,
            double _rWX, double _rWY, double _rWZ,
            double _cLH, double _cLW, double _cRH, double _cRW,
            bool _lC, bool _rC) //Alternate Constructor
    {
        gestureFound = _gF;
        leftFound = _lF;
        rightFound = _rF;
        gestureData = _gT;
        leftHand = xyz(_lHX, _lHY, _lHZ);
        leftWrist = xyz(_lWX, _lWY, _lWZ);
        rightHand = xyz(_rHX, _rHY, _rHZ);
        rightWrist = xyz(_rWX, _rWY, _rWZ);
        confLH = _cLH;
        confLW = _cLW;
        confRH = _cRH;
        confRW = _cRW;
        leftClick = _lC;
        rightClick = _rC;
    }

    // Are gesture data, left hand data, and right hand data valid
    bool gestureFound, leftFound, rightFound;

    // Type of gesture
    gestureType gestureData;

    // Data for each point
    xyz leftHand, rightHand, leftWrist, rightWrist;
    
    // Confidence for each point
    double confLH, confRH, confLW, confRW;
    
    // Rates of hand closure, clicks
    bool leftClick, rightClick;
} nuiData;

#endif /* NUITRACK_DATA_H */

