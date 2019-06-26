#ifndef NUITRACK_DATA_H
#define NUITRACK_DATA_H

#include <string>

using std::string;

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

typedef struct nuiData
{
    nuiData() //Default Constructor
    {
        gestureFound = false;
        leftFound = false;
        rightFound = false;
        gestureData = gestureType::NONE;
        leftWristX = 0.0;
        leftWristY = 0.0;
        leftWristZ = 0.0;
        leftHandX = 0.0;
        leftHandY = 0.0;
        leftHandZ = 0.0;
        rightWristX = 0.0;
        rightWristY = 0.0;
        rightWristZ = 0.0;
        rightHandX = 0.0;
        rightHandY = 0.0;
        rightHandZ = 0.0;
    }

    nuiData(bool _gF, bool _lF, bool _rF, gestureType _gT,
            double _lWX, double _lWY, double _lWZ,
            double _lHX, double _lHY, double _lHZ,
            double _rWX, double _rWY, double _rWZ,
            double _rHX, double _rHY, double _rHZ) //Alternate Constructor
    {
        gestureFound = _gF;
        leftFound = _lF;
        rightFound = _rF;
        gestureData = _gT;
        leftWristX = _lWX;
        leftWristY = _lWY;
        leftWristZ = _lWZ;
        leftHandX = _lHX;
        leftHandY = _lHY;
        leftHandZ = _lHZ;
        rightWristX = _rWX;
        rightWristY = _rWY;
        rightWristZ = _rWZ;
        rightHandX = _rHX;
        rightHandY = _rHY;
        rightHandZ = _rHZ;
    }

    // Are gesture data, left hand data, and right hand data valid
    bool gestureFound, leftFound, rightFound;

    // Type of gesture
    gestureType gestureData;

    // Data for each point
    double leftWristX, leftWristY, leftWristZ;
    double leftHandX, leftHandY, leftHandZ;
    double rightWristX, rightWristY, rightWristZ;
    double rightHandX, rightHandY, rightHandZ;
} nuiData;

#endif /* NUITRACK_DATA_H */

