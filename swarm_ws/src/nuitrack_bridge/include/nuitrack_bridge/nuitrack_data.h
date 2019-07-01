#ifndef NUITRACK_DATA_H
#define NUITRACK_DATA_H

#include <string>

using std::string;

// Physical transform of the Kinect's lens against global frame
#define KINECT_TRAN_X 0.113 //-45.0
#define KINECT_TRAN_Y -0.483 //-10.7
#define KINECT_TRAN_Z 0.557 //58.5
#define KINECT_QUAT_X 0.544 //0.543
#define KINECT_QUAT_Y 0.551 //0.550
#define KINECT_QUAT_Z 0.445 //0.444
#define KINECT_QUAT_W -0.451 //0.450

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
    }

    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gt,
            xyz _lH, xyz _lW, xyz _rH, xyz _rW) //Alternate Constructor
    {
        
    }
    
    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gT,
            double _lHX, double _lHY, double _lHZ,
            double _lWX, double _lWY, double _lWZ,
            double _rHX, double _rHY, double _rHZ,
            double _rWX, double _rWY, double _rWZ) //Alternate Constructor
    {
        gestureFound = _gF;
        leftFound = _lF;
        rightFound = _rF;
        gestureData = _gT;
        leftHand = xyz(_lHX, _lHY, _lHZ);
        leftWrist = xyz(_lWX, _lWY, _lWZ);
        rightHand = xyz(_rHX, _rHY, _rHZ);
        rightWrist = xyz(_rWX, _rWY, _rWZ);
    }

    // Are gesture data, left hand data, and right hand data valid
    bool gestureFound, leftFound, rightFound;

    // Type of gesture
    gestureType gestureData;

    // Data for each point
    xyz leftHand, rightHand, leftWrist, rightWrist;
} nuiData;

#endif /* NUITRACK_DATA_H */

