#include "arduino_server.h"
#include <sstream>
#include <string>
#include <ros/ros.h>
#include <ros/master.h>
#include <vicon_demo/rtheta.h>
#include <wvu_swarm_std_msgs/rtheta.h>

void commandCallback(command cmd)
{
    printf("Robot[%d] head %s\n", 0, "asdg");
}

void messageCallback(const wvu_swarm_std_msgs::rtheta &msg) {}

void info(const char *patt, void *dat)
{
    std::ostringstream os;
    os << "SERVER INFO: " << patt << "\n";
    std::string full_pattern = os.str();
    const char *ch_pathh = full_pattern.c_str();
    printf(ch_pathh, dat);
}

bool keepalive()
{
    return true;
}

void *sendThread(void *arg0)
{
    ros::Subscriber *sub = (ros::Subscriber*)arg0;
    
    sleep(1);
    while(sockets->size() <= 0) {usleep(10000);}
    
    puts("Sending messages");

    while (true)
    {
        wvu_swarm_std_msgs::rtheta vector =
                *(ros::topic::waitForMessage<wvu_swarm_std_msgs::rtheta>("/vicon_demo"));
        
        //command output = {"0,0.5,135.4"};
        command output;
        sprintf(output.str, "%d,%f,%f", 0, vector.radius, vector.degrees);
        
        sendCommandToRobots(output);
        usleep(10000);
    }
}

int main(int argc, char **argv)
{
    // Set up ros
    ros::init(argc, argv, "arduino_test");
    ros::NodeHandle n;
    ros::Subscriber vectorSub;
    ros::Rate rate(100);
    
    // Subscribe
    vectorSub = n.subscribe("/vicon_demo", 10, &messageCallback);
    
    puts("Starting");
    printf("Sending %d bytes per message\n", (int)sizeof(command));

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_t tid;
    pthread_create(&tid, &attr, sendThread, &vectorSub);

    beginServer(commandCallback, info, puts, keepalive);

    pthread_join(tid, NULL);
}