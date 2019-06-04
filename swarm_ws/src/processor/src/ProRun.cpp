#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.cpp"
#include <sstream>


void botCallback(const wvu_swarm_std_msgs::viconBotArray &msg)
{
  //For Debug purposes. callback should do nothing by default.
  //ROS_INFO("I hear: [%i]", msg.id1);
}

int main(int argc, char **argv)
{
  // Test code for neighbor finding
  Bot a(0, 0, "ab");
   Bot b(1, 1, "bc");
   Bot c(2, 2, "cd");
   Bot d(3, 3, "de");
   Bot e(4, 4, "ef");
   Bot f(5, 5, "fg");
   Bot g(6, 6, "gh");
   Bot h(7, 7, "hi");
   Bot i(8, 8, "ij");
   Bot j(9, 9, "jk");
   Bot inputList [10] = {a, b, c, d, e, f, g, h, i, j};
   Processor test_pros = Processor(inputList);
   test_pros.findNeighbors();
   test_pros.printBotMail();
   wvu_swarm_std_msgs::aliceMailArray msg = test_pros.createAliceMsg(9);
   test_pros.printAliceMail(msg);
   std::cout << std::endl;


  Processor bigbrain(0);
  ros::init(argc, argv, "Processor");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
  std::vector <ros::Publisher> pubVector;

  for (int i = 0; i < BOT_COUNT; i++) //Starts publishing to all 50 topics
  {
    ros::Publisher pub = n.advertise<wvu_swarm_std_msgs::aliceMailArray>("alice_mail_" + std::to_string(i), 1000);
    pubVector.push_back(pub);
  }

  while (ros::ok())
  {
    wvu_swarm_std_msgs::viconBotArray tempBotArray = *(ros::topic::waitForMessage<wvu_swarm_std_msgs::viconBotArray>
            ("vicon_array"));

    bigbrain.processVicon(tempBotArray);
    bigbrain.findNeighbors();
    // Bot a(0, 0, "ab");
    //  Bot b(1, 1, "bc");
    //  Bot c(2, 2, "cd");
    //  Bot d(3, 3, "de");
    //  Bot e(4, 4, "ef");
    //  Bot f(5, 5, "fg");
    //  Bot g(6, 6, "gh");
    //  Bot h(7, 7, "hi");
    //  Bot i(8, 8, "ij");
    //  Bot j(9, 9, "jk");
    //  Bot inputList [10] = {a, b, c, d, e, f, g, h, i, j};
    //  Processor test_pros = Processor(inputList);
    //  test_pros.findNeighbors();
     std::cout << "x"
    for (int i = 0; i < BOT_COUNT; i++) //Publishes msgs to Alices
    {
      pubVector.at(i).publish(bigbrain.createAliceMsg(i));
    }
    ros::spinOnce();
  }
  return 0;
}
