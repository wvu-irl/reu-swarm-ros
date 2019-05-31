#include "ros/ros.h"
#include "std_msgs/String.h"

#include <wvu_swarm_std_msgs/viconBotArray.h>

typedef struct
{
    int id;
    float xpos;
    float ypos;
    float heading;
} Bot;

typedef struct
{
    int id;
    //TODO
} Mail;

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
    Processor();
    void init();
    std::map<char*, int> inst();
    void processVicon(wvu_swarm_std_msgs::viconBotArray data);
  //  void start();
//    void stop();

  //  void findNeighbors;
  //  void findObject;

  //void publish_mail(const ros::TimerEvent&);

};

//#include "Processor.cpp"
