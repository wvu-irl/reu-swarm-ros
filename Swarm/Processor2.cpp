//
//  main.cpp
//  Processor
//
//  Created by Casey Edmonds-Estes on 5/30/19.
//  Copyright © 2019 Casey Edmonds-Estes. All rights reserved.
//

#include <iostream>
#include <math.h>

using namespace std;

struct Bot
{
    float x;
    float y;
    float distance = 10000; //Default value set arbitrarily for sorting purposes
    string name;
    Bot(float _x, float _y, string _name)
    {
        x = _x;
        y = _y;
        name = _name;
    }
    
    Bot()
    {
        x = 0;
        y = 0;
        name = "";
    }
};

bool compareTwoBots(Bot& a, Bot& b)
{
    return (float) a.distance > (float) b.distance; // Reverse the > to sort smallest first instead
}

static const int BOT_COUNT = 10; // Number of bots in array
static const int NEIGHBOR_COUNT = 4; // Number of neighbors desired

class Processor
{
    
    Bot bots [BOT_COUNT];
    Bot botMail [BOT_COUNT][NEIGHBOR_COUNT];
    
public:
    
    /*
     Takes an array and adds it to the processor
     This function will eventually be called by the listner
     */
    Processor(Bot _bots [])
    {
        for (int i = 0; i < BOT_COUNT; i++) {
            bots[i] = _bots[i];
        }
    }
    
    /*
     Prints the current array to send, for testing
     */
    void printBotMail()
    {
        for(int i = 0; i < BOT_COUNT; i++)
        {
            cout << "[";
            for(int j = 0; j < NEIGHBOR_COUNT; j++)
            {
                cout << " ";
                cout << botMail[i][j].name;
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
    void findNeighbors()
    {
        int botIndex = 0;
        for(auto& bot : bots)
        {
            int n = sizeof(botMail[botIndex])/sizeof(botMail[botIndex][0]); // for sorting later
            int curIndex = 0; // Becuase we're looping over the array, we have to track the index ourselves
            for(auto& cur : bots)
            {
                if (cur.name == bot.name) // Check for duplicates
                {
                    continue;
                }
                cur.distance = pow((cur.x - bot.x), 2) + pow((cur.y - bot.y), 2);
                bool smallest = true;
                sort(botMail[botIndex], botMail[botIndex]+n, compareTwoBots); // Sorts greatest first
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
};

int main() {
    Bot a(0, 0, "a");
    Bot b(1, 1, "b");
    Bot c(2, 2, "c");
    Bot d(3, 3, "d");
    Bot e(4, 4, "e");
    Bot f(5, 5, "f");
    Bot g(6, 6, "g");
    Bot h(7, 7, "h");
    Bot i(8, 8, "i");
    Bot j(9, 9, "j");
    Bot inputList [10] = {a, b, c, d, e, f, g, h, i, j};
    Processor test_pros = Processor(inputList);
    test_pros.findNeighbors();
    test_pros.printBotMail();
    return 0;
}

