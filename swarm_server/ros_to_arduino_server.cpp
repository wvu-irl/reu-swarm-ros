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

ros::Publisher g_from_ard; // global publisher for data from the arduinos

/**
 *  Callback for getting data from the arduino clients
 *  
 *  All recieved data will get published as soon as it comes
 */
void commandCallback(command cmd)
{
  ROS_INFO("Command: %s", cmd.str); // displaying recieved data
  server_setup::sensor_data inf; // conversion container
  inf.data = cmd.str; // copying data to message
  g_from_ard.publush(inf) // publishing
}

/**
 *  Info callback 
 * 
 *  Displays server information
 */
void info(const char *patt, void *dat)
{
  std::ostringstream os;
  os << "SERVER INFO: " << patt << "\n";
  std::string full_pattern = os.str();
  const char *ch_pathh = full_pattern.c_str();
  ROS_INFO(ch_pathh, dat);
}

/**
 *  callback for a subscription to send data to the swarm
 */
void sendToRobotCallback(server_setup::robotcommand msg)
{
  command cmd = {{'\0'}}; // creating command
  sprintf(cmd.str, "%f,%f", msg.r, msg.theta);
  char id[3] = {'\0'};
  strcpy(id, msg.rid.c_str()); // cpoying into message
  sendCommandToRobots(cmd, rid_map.at(id)); // sending to robots through TCP server
}

/**
 *  Callback to display server errors
 */
void errorCallBack(const char *msg)
{
  ROS_ERROR(msg);
}

/**
 *  Runs a loop separate from the server
 * 
 *  This is nessessary because the server contains a closed loop
 */
void *controlThread(void *arg0)
{
  ros::NodeHandle *n = (ros::NodeHandle *)arg0; // passed node handle
  ros::Subscriber to_ard = n->subscribe("execute", 1000, sendToRobotCallback); // subscribing to movment datastream
  g_from_ard = n->advertise<server_setup::sensor_data>("from_arduino", 1000); // advertising arduino data

  ros::spin(); // allowing subscriber callbacks to happen

  pthread_exit(0); // exiting thread
}

// main
int main(int argc, char **argv)
{
  ros::init(argc, argv, "arduino_server");
  ros::NodeHandle n;

  ROS_INFO("Starting");
  ROS_INFO("Sending %d bytes per message\n", (int)sizeof(command));

  // creating a separate thread for additional looped control
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_t tid;
  pthread_create(&tid, &attr, controlThread, &n);

  // starting server
  beginServer(commandCallback, info, errorCallBack, ros::ok);

  // waiting for thread to die
  pthread_join(tid, NULL);
}
