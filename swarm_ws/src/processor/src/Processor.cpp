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


std::pair<float,float> Processor::getSeparation(Bot _bot, std::pair<float,float> _obs, float _tolerance)//helper function for finding obstacle points
{// takes current
  float loc_r; //|distance| b/w bot and current obstacle point.
  float theta; //in radians
  float dx; //x separation.
  float dy; //y separation.
  
  dx = _obs.first - _bot.x;
  dy = _obs.second - _bot.y;


  loc_r = sqrt(pow(dx, 2) + pow(dy, 2)); //magnitude of separation

  if (loc_r<=_tolerance)
  {
    theta = tan(dx/dy)*(M_PI/180);
    std::pair<float,float> polar_point (loc_r, theta);
    return polar_point;
  }
  else
  {
    return *((std::pair<float, float> *) NULL);
  }
}

Processor::Processor(int a) //Default constructor, dummy parameter is there for compile reasons?
{
  {
    for (int i = 0; i < BOT_COUNT; i++)
    {
      bots[i] = Bot();
    }
  }
}

Processor::Processor(Bot _bots[]) //Constructor given a predetermined set of bots
{
  for (int i = 0; i < BOT_COUNT; i++)
  {
    bots[i] = _bots[i];
  }
}

void Processor::init()
{}


void Processor::processVicon(wvu_swarm_std_msgs::viconBotArray data) //Fills in bots[]
{
  for (size_t i = 0; i < sizeof(data.poseVect) / sizeof(data.poseVect[0]); i++)
  {
    char tempID[2] = {data.poseVect[i].botId[0], data.poseVect[i].botId[1]};
    size_t numID = rid_map[tempID];

    // the incoming geometry_msgs::Quaternion is transformed to a tf::Quaterion
    tf::Quaternion quat;
    tf::quaternionMsgToTF(data.poseVect[i].botPose.transform.rotation, quat);

    // the tf::Quaternion has a method to access roll pitch and yaw (yaw is all we need in a 2D plane)
    double roll, pitch, yaw;
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    bots[numID] = Bot(tempID, data.poseVect[i].botPose.transform.translation.x,
                      data.poseVect[i].botPose.transform.translation.y, yaw, 10000);
  }
}

void Processor::printBotMail() //Prints the id's in botMail[] to the console
{
  using namespace std;
  for (int i = 0; i < BOT_COUNT; i++)
  {
    std::cout << "[";
    for (int j = 0; j < NEIGHBOR_COUNT; j++)
    {
      cout << " ";
      cout << botMail[i][j].id[0] << botMail[i][j].id[1];
      cout << " ";
    }
    cout << "]\n";
  }
}

void Processor::printAliceMail(wvu_swarm_std_msgs::aliceMailArray _msg)
{
  using namespace std;

    std::cout << "[";
    for (int j = 0; j < NEIGHBOR_COUNT; j++)
    {
      cout << " ";
      cout << _msg.aliceMail[j].theta;
      cout << " ";
    }
    cout << "]\n";

}

/*
     Exactly what it sounds like
     This function finds the nearest few neighbors
     The number of neighbors can be set by NEIGHBOR_COUNT
     */
void Processor::findNeighbors()
{
  int botIndex = 0;
  int j; //iterator for obs finding loop
  int num_pts = static_cast<int>(obs.size());
<<<<<<< HEAD
<<<<<<< Upstream, based on master

=======
  
>>>>>>> da4f3dc updated obstacles searching. 
=======
  
>>>>>>> refs/remotes/origin/Processor-test
  std::pair<float,float> new_pair; //holds the return pair
  float tolerance = 12; //this value is supposed to be our actual tolerance (just made it 12).
  //holds obs near a robot for each robot.
  //probably not our long term solution for storing the data.

  for (auto &bot : bots)
  {
    j = 0;
    while (j < num_pts) //runs for each point in the obs
    {
      std::pair<float,float> iter_obs = obs.at(j); //current obs to test
      new_pair = getSeparation(bot, iter_obs, tolerance);
      if (&new_pair != NULL)
      {
        polar_obs[j].push_back(new_pair);
      }
      j++;
    }

    //int n = sizeof(botMail[botIndex]) / sizeof(botMail[botIndex][0]); // for sorting later
    int curIndex = 0; // Because we're looping over the array, we have to track the index ourselves
    for (auto &cur : bots)
    {
      if (cur.id[0] == bot.id[0] && cur.id[1] == bot.id[1]) // Check for duplicates
      {
        continue;
      }
      cur.distance = sqrt(pow((cur.x - bot.x), 2) + pow((cur.y - bot.y), 2));
      bool smallest = true;

      for (int i = 0; i < NEIGHBOR_COUNT; i++)
      {
        if (cur.distance > botMail[botIndex][i].distance)
        {
          if (i == 0) // If cur is further than the first in the array, it's further
          {
            smallest = false;
            break;
          } else
          { // This means cur is neither the furthest nor nearest
            botMail[botIndex][0] = cur;
            smallest = false;
            std::sort(botMail[botIndex], botMail[botIndex] + NEIGHBOR_COUNT,
                      compareTwoBots); // Sorts greatest first
            break;
          }
        }
      }
      if (smallest)
      { // If cur is closer than everything, it's the closest
        botMail[botIndex][0] = cur;
        std::sort(botMail[botIndex], botMail[botIndex] + NEIGHBOR_COUNT,
                  compareTwoBots); // Sorts greatest first
      }
      curIndex++;
    }

    botIndex++;
  }
}

wvu_swarm_std_msgs::aliceMailArray Processor::createAliceMsg(int i) //Turns information to be sent to Alice into a msg
{
  wvu_swarm_std_msgs::aliceMailArray _aliceMailArray;
  for (int j = 0; j < NEIGHBOR_COUNT; j++) //Transfers fields of the struct to fields of the msg
  {

    if (botMail[i][j].y-bots[i].y > 0) {
      _aliceMailArray.aliceMail[j].theta = fmod(atan((botMail[i][j].y-bots[i].y)/(botMail[i][j].x-bots[i].x))-M_PI_2-bots[i].heading,2*M_PI);
    }
    else {
      _aliceMailArray.aliceMail[j].theta = fmod(atan((botMail[i][j].y-bots[i].y)/(botMail[i][j].x-bots[i].x))+M_PI_2-bots[i].heading,2*M_PI);
    }
    _aliceMailArray.aliceMail[j].distance = botMail[i][j].distance;
    _aliceMailArray.aliceMail[j].heading = fmod(botMail[i][j].heading-bots[i].heading,2*M_PI);
  }
  return _aliceMailArray;

}
