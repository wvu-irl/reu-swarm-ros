#include <iostream>
#include <stdio.h>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>

void msgCallback(const visualization_msgs::MarkerArray &msg) {}

// Find x,y where the line passing between alpha and beta intercepts an xy plane at z
geometry_msgs::Point findZIntercept(geometry_msgs::Point _alpha,
        geometry_msgs::Point _beta, double _zed)
{
    /* THEORY
     * Equation of line: r(t) = v*t+v0
     * Direction vector: v = (xa - xb, ya - yb, za - zb)
     * Offset vector: v0 = (xa, ya, za)
     * Plug in z to r(t), solve for t, use t to solve for x and y/
     */
    
    geometry_msgs::Point ret;
    
    // Check if no solution
    if(_alpha.z == _beta.z) {
        printf("\033[1;31mhand_pointer: \033[0;31mNo solution for intercept\033[0m\n");
        ret.x = 0.0;
        ret.y = 0.0;
        ret.z = 0.0;
    }
    else {
        double t = (_zed - _alpha.z) / (_alpha.z - _beta.z);
        double x = _alpha.x * (t + 1) - _beta.x * t;
        double y = _alpha.y * (t + 1) - _beta.y * t;

        ret.x = x;
        ret.y = y;
        ret.z = _zed;
    }
    
    return ret;
}

int main(int argc, char** argv) {
    // ROS setup
    ros::init(argc, argv, "hand_pointer");
    
    // Generates nodehandles, publishers, subscriber
    ros::NodeHandle n;
    ros::NodeHandle n_priv("~"); // private handle
    ros::Publisher pub1, pub2, pub3;
    ros::Subscriber sub;
    pub1 = n.advertise<geometry_msgs::Point>("nuitrack_bridge/hand_1", 1000);
    pub2 = n.advertise<geometry_msgs::Point>("nuitrack_bridge/hand_2", 1000);
    sub = n.subscribe("nuitrack_bridge", 10, &msgCallback);
    
    // Create two hand pointers for output and a markerarray for input
    geometry_msgs::Point handOne, handTwo;
    visualization_msgs::MarkerArray mArr;
    
    while(ros::ok()){
        mArr = *(ros::topic::waitForMessage<visualization_msgs::MarkerArray>("nuitrack_bridge"));
        
        // Create data to hold joints to find and bools of whether they're found
        geometry_msgs::Point lH, lW, rH, rW;
        bool foundLH, foundLW, foundRH, foundRW;
        
        // Iterate through received data, look for our points
        for(visualization_msgs::Marker m : mArr.markers)
        {
            // 1 = lH, 2 = lW, 3 = rH, 4 = rW
            if(m.id == 1) {
                foundLH = true;
                lH.x = m.pose.position.x;
                lH.y = m.pose.position.y;
                lH.z = m.pose.position.z;
            }
            else if(m.id == 2) {
                foundLW = true;
                lW.x = m.pose.position.x;
                lW.y = m.pose.position.y;
                lW.z = m.pose.position.z;
            }
            else if(m.id == 3) {
                foundRH = true;
                rH.x = m.pose.position.x;
                rH.y = m.pose.position.y;
                rH.z = m.pose.position.z;
            }
            else if(m.id == 4) {
                foundRW = true;
                rW.x = m.pose.position.x;
                rW.y = m.pose.position.y;
                rW.z = m.pose.position.z;
            }
        }
        
        // Move the left hand pointer if joints were found
        if(foundLH && foundLW) {
            handOne = findZIntercept(lH, lW, 0);
        }
        
        // Move the right hand pointer if joints were found
        if(foundRH && foundRH) {
            handTwo = findZIntercept(rH, rW, 0);
        }
        
        // Publish hands
        pub1.publish(handOne);
        pub2.publish(handTwo);
    }
    
    return 0;
}

