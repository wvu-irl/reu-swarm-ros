#include <ros/ros.h>
#include <server_setup/robotcommand.h>
#include <server_setup/sensor_data.h>

#include "arduino_server.h"
#include "robot_id.h"
#include <stdlib.h>
#include <string.h>
#include <map>
#include <sstream>
#include <string>
#include <functional>

std::vector<command> cmd_queue;

void commandCallback(command cmd)
{
    ROS_INFO("Command: %s", cmd.str);
    cmd_queue.push_back(cmd);
}

void info(const char *patt, void *dat)
{
    std::ostringstream os;
    os << "SERVER INFO: " << patt << "\n";
    std::string full_pattern = os.str();
    const char *ch_pathh = full_pattern.c_str();
    ROS_INFO(ch_pathh, dat);
}

void sendToRobotCallback(server_setup::robotcommand msg)
{
  command cmd = {{'\0'}};
  sprintf(cmd.str, "%f,%f", msg.r, msg.theta);
  char id[3] = {'\0'};
  strcpy(id, msg.rid.c_str());
  sendCommandToRobots(cmd, rid_map.at(id));
}

void errorCallBack(const char *msg)
{
  ROS_ERROR(msg);
}

void *sendThread(void *arg0)
{
    ros::NodeHandle *n = (ros::NodeHandle *)arg0;

    // find actual topic to subscribe to
    ros::Subscriber to_ard = n->subscribe("to_ard", 1000, sendToRobotCallback);

    ros::Publisher from_ard = n->advertise<server_setup::sensor_data>("from_arduino", 1000);

    while (ros::ok())
    {
      while (cmd_queue.size() > 0)
      {
        server_setup::sensor_data pub_command;
        pub_command.data = cmd_queue.at(0).str;
        cmd_queue.erase(cmd_queue.begin());
        from_ard.publish(pub_command);
      }
      ros::spinOnce();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arduino_server");
    ros::NodeHandle n;

    ROS_INFO("Starting");
    ROS_INFO("Sending %d bytes per message\n", (int)sizeof(command));

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_t tid;
    pthread_create(&tid, &attr, sendThread, &n);

    beginServer(commandCallback, info, errorCallBack, ros::ok);

    pthread_join(tid, NULL);
}
