#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.h"
#include <stdlib.h>
#include <map>







Processor::Processor(){}

void Processor::init(){}

// creating a map for all the string to integer values
std::map<char*, int> Processor::inst()
{
  char *LETTERARRAY[] = {"DE", "PA", "NJ", "GA", "CT", "MA", "MD", "SC", "NH", "VA", "NY", "NC",
                                  "RI", "VT", "KY", "TN", "OH", "LA", "IN", "MS", "IL", "AL", "ME", "MO",
                                  "AR", "MI", "FL", "TX", "IA", "WI", "CA", "MN", "OR", "KA", "WV", "NV",
                                  "NE", "CO", "ND", "SD", "MT", "WA", "ID", "WY", "UT", "OK", "NM", "AZ",
                                  "AK", "HI"};
  std::map<char*, int> map;
  for (size_t i = 0; i < sizeof(LETTERARRAY) / sizeof(LETTERARRAY[0]); i++)
  {
    map.insert(std::pair<char*, size_t>(LETTERARRAY[i], i));
  }
  return map;
}

void Processor::processVicon(wvu_swarm_std_msgs::viconBotArray data){


}
