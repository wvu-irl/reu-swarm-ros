#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <aruco_markers/Marker.h>
#include <aruco_markers/MarkerArray.h>
#include <ros/console.h>
#include <iostream>
#include <geometry_msgs/Twist.h>
#include <cmath>
//aruco_markers::MarkerArray marker_list;

void arucoCallback(const aruco_markers::MarkerArray::ConstPtr &msg)
{
}

int main(int argc, char** argv){

//initializes the node named "Robot_GP"
ros::init(argc,argv,"aruco_positioning_node");
// Generates an object named n that interacts with ROS functions
ros::NodeHandle n,nh;
ros::Publisher robotGP_pub;
ros::Subscriber aruco_sub;
//sets the looprate at 10 Hz with looprate sleep it will never go faster than this but can go slower
ros::Rate loop_rate(10);


//we have a aruco_marker  that subscribes to a message topic called joy
aruco_sub = n.subscribe<aruco_markers::MarkerArray>("/markers", 10, &arucoCallback);
robotGP_pub = nh.advertise<geometry_msgs::Twist>("robotGP_pub", 1000);

//this is just saying while ros is running execute
while (ros::ok()){
  aruco_markers::MarkerArray marker_list= *(ros::topic::waitForMessage<aruco_markers::MarkerArray>("/markers"));


geometry_msgs::Twist distance;
int markerID;
markerID=marker_list.markers[0].id;
distance.linear.x=marker_list.markers[0].tvec.z;
distance.linear.y=marker_list.markers[0].tvec.x;
distance.angular.x=marker_list.markers[0].rpy.x;
distance.angular.y=marker_list.markers[0].rpy.y;
distance.angular.z=marker_list.markers[0].rpy.z;

if (markerID ==0) {
/*
  distance.linear.x=172-floor(distance.linear.x);
  distance.linear.y=1041-floor(distance.linear.y);
  distance.angular.z=90-floor(distance.angular.z);
*/
  distance.linear.x=172-(distance.linear.x);
  distance.linear.y=1041-(distance.linear.y);
  distance.angular.y=(distance.angular.y);
}else if (markerID ==1) {
  distance.linear.x=20+(distance.linear.x);
  distance.linear.y=1285+(distance.linear.y);
  distance.angular.y=(distance.angular.y)-M_PI;
}else if (markerID ==2) {
  distance.linear.x=172-(distance.linear.x);
  distance.linear.y=1468-(distance.linear.y);
  distance.angular.y=(distance.angular.y);
}


robotGP_pub.publish(distance);

}
loop_rate.sleep();
ros::spinOnce();
}
