#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.cpp"
#include <sstream>




void botCallback(const wvu_swarm_std_msgs::viconBotArray& msg)
{
  //For Debug purposes. callback should do nothing by default.
  //ROS_INFO("I hear: [%i]", msg.id1);
}

// void botCallback(const std_msgs::String::ConstPtr& msg)
// {
//    ROS_INFO("I hear: [%s]", msg->data.c_str());
// }


int main(int argc, char **argv){

    Processor bigbrain(0);
    ros::init(argc, argv, "Processor");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("BotStructs", 1000, botCallback);
    ros::Publisher pub = n.advertise<std_msgs::String>("MailStructs", 1000);
    while(true){
        std::cout << "yo we here";
        wvu_swarm_std_msgs::viconBotArray tempBotArray =
        *(ros::topic::waitForMessage<wvu_swarm_std_msgs::viconBotArray>("/ViconArray"));

        bigbrain.processVicon(tempBotArray);
    }



    // std_msgs::String str;
    // str.data = "hello world";
    // std::cout << "yo";
    // pub.publish(str);
    // ros::spin();
    return 0;
}
