#include "ros/ros.h"
#include "std_msgs/String.h"

#include <wvu_swarm_std_msgs/viconBotArray.h>

typedef struct Bot
{
    Bot(){
      id[0]='\0';
      id[1]='\0';
      xpos =0;
      ypos=0;
      heading=0;
    }

    Bot(char _id[2], float _xpos, float _ypos, float _heading) {
      id[0]=_id[0];
      id[1]=_id[1];
      xpos =_xpos;
      ypos=_ypos;
      heading=_heading;
    }

    char id[2];
    float xpos;
    float ypos;
    float heading;
} Bot;

typedef struct
{
    int id;
    //TODO
} Mail;

// creating a map for all the string to integer values
std::map<char*, int> inst() {
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
std::map<char*, int> rid_map = inst();

class Processor {

  private:
      Bot prevBotArray [50];
      Bot curBotArray [50];
      Mail mailbox [50];
      std::vector<std::pair <float,float> > obs;

      ros::Timer timer;
      ros::Publisher pub;
      ros::Subscriber sub;

  public:
    Processor(int a);
    void init();
    void processVicon(wvu_swarm_std_msgs::viconBotArray data);
  //  void start();
//    void stop();

  //  void findNeighbors;
  //  void findObject;

  //void publish_mail(const ros::TimerEvent&);

};

//#include "Processor.cpp"
