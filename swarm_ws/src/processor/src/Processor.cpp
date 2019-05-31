#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.h"
#include <stdlib.h>
#include <map>
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <tf/LinearMath/Matrix3x3.h>
#include <math.h>




Processor::Processor(int a){}

void Processor::init(){}



void Processor::processVicon(wvu_swarm_std_msgs::viconBotArray data){

    for(size_t i = 0; i< sizeof(data.poseVect)/ sizeof(data.poseVect[0]); i++)
    {
        char tempID[2] = {data.poseVect[i].botId[0],data.poseVect[i].botId[1]};
        size_t numID = rid_map[tempID];

        // the incoming geometry_msgs::Quaternion is transformed to a tf::Quaterion
        tf::Quaternion quat;
        tf::quaternionMsgToTF(data.poseVect[i].botPose.transform.rotation, quat);

    // the tf::Quaternion has a method to acess roll pitch and yaw
        double roll, pitch, yaw;
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        curBotArray[numID]=Bot(tempID,data.poseVect[i].botPose.transform.translation.x,
          data.poseVect[i].botPose.transform.translation.y,yaw);
    } 
}
