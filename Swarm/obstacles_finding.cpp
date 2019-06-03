/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   obstacles_handeling.cpp
 * Author: Henry Gunner
 *
 * Created on May 30, 2019, 4:21 PM
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

class Obstacles_Finding
{ 
struct Robot
{
    float x;
    float y;
    float speed;
    float angle;
    float time_stamp;
    string id; 
    
};

struct Obstacle
{ //has two vectors, which concurrently house a point cloud
    std::vector <float> x;
    std::vector <float> y; 
    //Obstacle()
};



float getSeperation(Robot bot, Obstacle obs, int size)
{
    //obs positions are vector<float[2]> types. 
    int i = 0;
    float r_min;
    float loc_r;

    float dx;
    float dy;

    
    while(i < size) //runs for each point in an obstacles point cloud
    {
        dx = obs.x.at(0)-bot.x; //x separation.
        dy = obs.y.at(0)-bot.y; //y separation.
        
        /*
        std::cout<<"obs.x.at(0)="<< obs.x.at(0) << "; ";
        std::cout<<"obs.y.at(0)="<< obs.y.at(0)<<endl;
        
        std::cout<<"bot.x="<< bot.x << "; ";
        std::cout<<"bot.y="<< bot.y<<endl;
        
        std::cout<<"dx= "<< dx << "; ";
        std::cout<<"dy= "<< dy;
        std::cout<< "------------------------------------------------\n";
         */
        
        loc_r = sqrt(pow(dx,2)+pow(dy,2)); //magnitude of separation 

        if (i==0)
        {
            r_min = loc_r;
        }
        else
        {
            if (loc_r < r_min)
            { 
                r_min = loc_r;
            }
        }
        i++;
    }
    return r_min;
}

Robot testRobSet(int size) //might need to declare a definate int value for array sizes
{
    int i = 0;
    float x;
    float y;
    
    //srand(134);
    if (size == 1)
    {
        Robot rob = {1,2,3,4,5,"fuck_my_life"};
        return rob;
    }
    
    if (size == 2)
    {
        Robot rob = {6,7,8,9,10,"fuck_you"};
        return rob;
    }
}

Obstacle testObsSet()
{
    int i = 0;
    float x;
    float y;
    
    std::vector <float> vx;
    std::vector <float> vy;
    float pos1[2];

    x = 10; //rand()%100;
    y = 10; //rand()%100;
    pos1[0] = x;
    pos1[1] = y;
    
    vx.push_back(x);
    vy.push_back(y);
    Obstacle o = {vx,vy};
    
    
    return o; 
}
 

void obstaclesFinding()
{
    const int num_bots = 2; //number of robots in the sim/system
    const int num_obs = 2; //dont make this larger than num_bots
    const float tolerance = 14; //tolerance for bots "seeing" things (obs/bots).
    
    float dist_to_obs; //distance between a bot and the closest part of an obs. 
    
    //these 4 objects are purely for testing purposes, to represent input data.
    //*
    Obstacle obs [num_obs] = {testObsSet(), testObsSet()};
    Obstacle neighbor_obs [num_obs][num_obs]; 
    Robot fake_prev_bots [num_bots]; 
    Robot fake_curr_bots [num_bots] = {testRobSet(1), testRobSet(2)};
    //*
    
    Robot check_bot;  //used as place holder in 1st while loop for each robot.
    Obstacle iter_obs; //used as place holder in 2nd while loop for each obstacle.
    
    int i = 0; //outer iterator
    int j = 0; //inner iterator
    
    i = 0;
    while (i < num_bots) //will have to be 50 (the number of robots)
    {   
        j = 0;
        check_bot = fake_curr_bots[i];
        while (j < num_obs) //will have to be the number of robots again.
        {
            iter_obs = obs[j]; //current obs to test
            if (&iter_obs != NULL)
            {
                dist_to_obs = getSeperation(check_bot, iter_obs, num_obs);
                if(dist_to_obs<=tolerance) 
                {
                    neighbor_obs[i][j] = iter_obs;
                }   
            }
            j++;
        }
        i++;
    }   
}
  

};    


