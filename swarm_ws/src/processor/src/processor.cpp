/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   processor.cpp
 * Author: smart3
 * 
 * Created on June 3, 2019, 10:34 AM
 */

#include "processor.h"
//#include "ros/ros.h"
//#include "std_msgs/String.h"
//#include "Processor.h"
#include <stdlib.h>
#include <map>
//#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <tf/LinearMath/Matrix3x3.h>
#include <math.h>




Processor::Processor(int a){
    {
      for (int i = 0; i < BOT_COUNT; i++) {
          bots[i] = Bot();
      }
    }
}

Processor::Processor(Bot _bots [])
    {
        for (int i = 0; i < BOT_COUNT; i++) {
            bots[i] = _bots[i];
        }
    }

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
        bots[numID]=Bot(tempID,data.poseVect[i].botPose.transform.translation.x,
          data.poseVect[i].botPose.transform.translation.y,yaw,10000);
    }
}

void Processor::printBotMail()
{
    using namespace std;
    for(int i = 0; i < BOT_COUNT; i++)
    {
        std::cout << "[";
        for(int j = 0; j < NEIGHBOR_COUNT; j++)
        {
            cout << " ";
            cout << botMail[i][j].id[0] << botMail[i][j].id[1];
            cout << " ";
        }
        cout << "]\n";
    }
}

/*
     Exactly what it sounds like
     This function finds the nearest few neighbors
     The number of neighbors can be set by NEIGHBOR_COUNT
     */
    void Processor::findNeighbors()
    {
        int botIndex = 0;
        int num_obs = sizeof(obs) 
        
        for(auto& bot : bots)
        {
            
            j = 0;
            while (j < num_obs) // 
            {
                iter_obs = obs[j]; //current obs to test
                if (&iter_obs != NULL)
                {
                    dist_to_obs = getSeperation(bot, iter_obs);
                    if(dist_to_obs<=tolerance) 
                    {
                        neighbor_obs[i][j] = iter_obs;
                    }   
                }
                j++;
            } 
            
            int n = sizeof(botMail[botIndex])/sizeof(botMail[botIndex][0]); // for sorting later
            int curIndex = 0; // Becuase we're looping over the array, we have to track the index ourselves
            for(auto& cur : bots)
            {
                if (cur.id[0] == bot.id[0] && cur.id[1] == bot.id[1]) // Check for duplicates
                {
                    continue;
                }
                cur.distance = pow((cur.x - bot.x), 2) + pow((cur.y - bot.y), 2);
                bool smallest = true;
                std::sort(botMail[botIndex], botMail[botIndex]+n, compareTwoBots); // Sorts greatest first
                for (int i = 0; i < NEIGHBOR_COUNT; i++)
                {
                    if (cur.distance > botMail[botIndex][i].distance)
                    {
                        if (i == 0) // If cur is further than the first in the array, it's further
                        {
                            smallest = false;
                            break;
                        } else { // This means cur is neither the furthest nor nearest
                            botMail[botIndex][0] = cur;
                            smallest = false;
                            break;
                        }
                    }
                }
                if (smallest) { // If cur is closer than everything, it's the closest
                    botMail[botIndex][0] = cur;
                }
                curIndex++;
            }
            botIndex++;
        }
    }