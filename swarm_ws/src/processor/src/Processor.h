#include "ros/ros.h"
#include "std_msgs/String.h"
#include <wvu_swarm_std_msgs/aliceMail.h>
#include <wvu_swarm_std_msgs/aliceMailArray.h>
#include <wvu_swarm_std_msgs/viconBotArray.h>

typedef struct Bot
{
    Bot(){
      id[0]='\0';
      id[1]='\0';
      x =0;
      y=0;
      heading=0;
      distance = 10000;
    }

    Bot(float _x, float _y,std::string name){
      x =_x;
      y=_y;
      strcpy(id,name.c_str());
      heading=0;
      distance=10000;
    }

    Bot(char _id[2], float _x, float _y, float _heading, float _distance) {
      id[0]=_id[0];
      id[1]=_id[1];
      x =_x;
      y=_y;
      heading=_heading;
      distance=_distance;
    }

    char id[2];
    float x;
    float y;
    float heading;
    float distance;
} Bot;

//----------------new stuff add for Obstacle finding ----------------
typedef struct Obstacle
{ //has two vectors, which concurrently house a point cloud.
    Obstacle(std::vector <float> _x, std::vector <float> _y){
        std::vector <float> x = _x;
        std::vector <float> y = _y; 
    }
    //Obstacle()
} Obstacle;
//-----------------------------------------------------



typedef struct
{
    int id;
    //TODO
} Mail;

bool compareTwoBots(Bot& a, Bot& b)
{
    return (float) a.distance > (float) b.distance; // Reverse the > to sort smallest first instead
}

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




static const int BOT_COUNT = 10; // Number of bots in array
static const int NEIGHBOR_COUNT = 4; // Number of neighbors desired

class Processor {

  private:
      Bot bots [BOT_COUNT];
      Bot botMail [BOT_COUNT][NEIGHBOR_COUNT];
      std::vector<std::pair <float,float> > obs;

      ros::Timer timer;
      ros::Publisher pub;
      ros::Subscriber sub;
  public:
      Processor(int a);
      Processor(Bot _bots []);
      void init();
      void processVicon(wvu_swarm_std_msgs::viconBotArray data);
      void printBotMail();
      void findNeighbors();
      wvu_swarm_std_msgs::aliceMailArray createAliceMsg(int i);
    
      float getSeperation(Bot _bot, Obstacle _obs); //finds sep b/w closest point of _obs and given _bot
    
  //  void start();
//    void stop();

  //  void findNeighbors;
  //  void findObject;

  //void publish_mail(const ros::TimerEvent&);

};

//#include "Processor.cpp"
