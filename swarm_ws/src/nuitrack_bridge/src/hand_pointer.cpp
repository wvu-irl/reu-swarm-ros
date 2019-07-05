#include <iostream>
#include <stdio.h>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>

void msgCallback(const visualization_msgs::MarkerArray &msg) {}

// Find the closest points between q(t), a vector pointing from _alpha to _beta,
//   and u(t), a vector pointing normal to the xy plane intersecting _source
geometry_msgs::Point findZHeight(geometry_msgs::Point _alpha,
        geometry_msgs::Point _beta, geometry_msgs::Point _source)
{
    /* THEORY
     * Equation of line: q(t) = v*t + v0
     * Direction vector: v = (xa - xb, ya - yb, za - zb)
     * Offset vector: v0 = (xa, ya, za)
     * 
     * Equation of line: u(t) = khat*t + u0
     * Offset vector: u0 = (xu, yu, 0)
     * 
     * Let f(t) = ||q(t) - u(t)||^2
     * Find f(t)
     * Find f'(t)
     * Solve for f'(t) = 0
     * Plug solution into u(t)
     */
    
    geometry_msgs::Point ret;
    
    // Check if infinite solutions
    if(_alpha.x == _beta.x && _alpha.y == _beta.y) {
        printf("\033[1;31mhand_pointer: \033[0;31mNo solution for height\033[0m\n");
        ret.x = 0.0;
        ret.y = 0.0;
        ret.z = 0.0;
    }
    else {
        // Get xv, yv, zv
        double x_v = _alpha.x - _beta.x;
        double y_v = _alpha.y - _beta.y;
        double z_v = _alpha.z - _beta.z;

        // Solve for t in f'(t) = 0
        double t = ((x_v * _source.x) + (y_v * _source.y) + ((1 - z_v) * _alpha.z)
                    - (_alpha.x * x_v) - (_alpha.y * y_v)) / (x_v*x_v + y_v*y_v - 2*z_v);
        
        // Plug into u(t)
        ret.x = _source.x;
        ret.y = _source.y;
        ret.z = t;
    }
    
    return ret;
}

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
    pub3 = n.advertise<visualization_msgs::MarkerArray>("test", 1000);
    sub = n.subscribe("nuitrack_bridge", 10, &msgCallback);
    
    // Create two hand pointers for output and a markerarray for input
    geometry_msgs::Point handOne, handTwo;
    visualization_msgs::MarkerArray mArr, testArr;
    
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
        
        geometry_msgs::Point testOrigin, testResult;
        testOrigin.x = 0; testOrigin.y = 0; testOrigin.z = 0;
        
        if(foundLH && foundLW) {
            testResult = findZHeight(lH, lW, testOrigin);
            
            visualization_msgs::Marker ret;
            ret.header.stamp = ros::Time();
            ret.header.frame_id = "world";
            ret.ns = "hand";
            ret.id = 5;
            ret.type = visualization_msgs::Marker::SPHERE;
            ret.pose.position.x = testResult.x;
            ret.pose.position.y = testResult.y;
            ret.pose.position.z = testResult.z;
            ret.pose.orientation.x = 0;
            ret.pose.orientation.y = 0;
            ret.pose.orientation.z = 0;
            ret.pose.orientation.w = 1;
            ret.scale.x = 0.04;
            ret.scale.y = 0.04;
            ret.scale.z = 0.04;
            ret.color.r = 1;
            ret.color.g = 1;
            ret.color.b = 0;
            ret.color.a = 1;
            
            testArr.markers.clear();
            testArr.markers.push_back(ret);
        }
        
        // Publish hands
        pub1.publish(handOne);
        pub2.publish(handTwo);
        pub3.publish(testArr);
    }
    
    return 0;
}

