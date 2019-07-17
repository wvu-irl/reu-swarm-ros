#include <iostream>
#include <stdio.h>
#include <list>

// ROS includes
#include <ros/ros.h>
#include <nuitrack_bridge/nuitrack_data.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <wvu_swarm_std_msgs/nuitrack_data.h>

void msgCallback(const wvu_swarm_std_msgs::nuitrack_data &msg) {
}

// Function to add together two nuitrackData messages
wvu_swarm_std_msgs::nuitrack_data combineNuiData(wvu_swarm_std_msgs::nuitrack_data _first,
        wvu_swarm_std_msgs::nuitrack_data _second) {
    wvu_swarm_std_msgs::nuitrack_data ret;

    // They didn't overload + for Point messages -_-
    ret.leftWrist.x = _first.leftWrist.x + _second.leftWrist.x;
    ret.leftWrist.y = _first.leftWrist.y + _second.leftWrist.y;
    ret.leftWrist.z = _first.leftWrist.z + _second.leftWrist.z;

    ret.leftHand.x = _first.leftHand.x + _second.leftHand.x;
    ret.leftHand.y = _first.leftHand.y + _second.leftHand.y;
    ret.leftHand.z = _first.leftHand.z + _second.leftHand.z;

    ret.rightWrist.x = _first.rightWrist.x + _second.rightWrist.x;
    ret.rightWrist.y = _first.rightWrist.y + _second.rightWrist.y;
    ret.rightWrist.z = _first.rightWrist.z + _second.rightWrist.z;

    ret.rightHand.x = _first.rightHand.x + _second.rightHand.x;
    ret.rightHand.y = _first.rightHand.y + _second.rightHand.y;
    ret.rightHand.z = _first.rightHand.z + _second.rightHand.z;

    ret.leftClick = _first.leftClick || _second.leftClick;
    ret.rightClick = _first.rightClick || _second.rightClick;

    if (_first.gestureData != (char)gestureType::NONE) {
        ret.gestureData = _first.gestureData;
    } else if (_second.gestureData != (char)gestureType::NONE) {
        ret.gestureData = _second.gestureData;
    } else {
        ret.gestureData = (char)gestureType::NONE;
    }

    return ret;
}

// Function to divide an nuitrackData by a scalar
wvu_swarm_std_msgs::nuitrack_data divideNuiData(wvu_swarm_std_msgs::nuitrack_data _nui, double _denom) {
    wvu_swarm_std_msgs::nuitrack_data ret;

    // They didn't overload + for Point messages -_-
    ret.leftWrist.x = _nui.leftWrist.x / _denom;
    ret.leftWrist.y = _nui.leftWrist.y / _denom;
    ret.leftWrist.z = _nui.leftWrist.z / _denom;

    ret.leftHand.x = _nui.leftHand.x / _denom;
    ret.leftHand.y = _nui.leftHand.y / _denom;
    ret.leftHand.z = _nui.leftHand.z / _denom;

    ret.rightWrist.x = _nui.rightWrist.x / _denom;
    ret.rightWrist.y = _nui.rightWrist.y / _denom;
    ret.rightWrist.z = _nui.rightWrist.z / _denom;

    ret.rightHand.x = _nui.rightHand.x / _denom;
    ret.rightHand.y = _nui.rightHand.y / _denom;
    ret.rightHand.z = _nui.rightHand.z / _denom;

    ret.leftClick = _nui.leftClick;
    ret.rightClick = _nui.rightClick;
    ret.gestureData = _nui.gestureData;

    return ret;
}

// Generates a visualizable Marker from a Point
visualization_msgs::Marker pointToMarker(int _id, geometry_msgs::Point _pt, double _r, double _g, double _b, double _a = 1.0, double _size = 4)
{
    visualization_msgs::Marker ret;

    ret.header.stamp = ros::Time(); // Metadata
    ret.header.frame_id = "world";
    ret.ns = "nui_smoothed";
    ret.id = _id; // Unique identifier
    ret.type = visualization_msgs::Marker::SPHERE;
    ret.pose.position.x = _pt.x; // Measured in meters
    ret.pose.position.y = _pt.y;
    ret.pose.position.z = _pt.z;
    ret.pose.orientation.x = 0; // Quaternion
    ret.pose.orientation.y = 0;
    ret.pose.orientation.z = 0;
    ret.pose.orientation.w = 1;
    ret.scale.x = _size; // Measured in meters
    ret.scale.y = _size;
    ret.scale.z = _size;
    ret.color.r = _r;
    ret.color.g = _g;
    ret.color.b = _b;
    ret.color.a = _a;

    return ret;
}

// Converts an nuitrack message to visualizable markers
visualization_msgs::MarkerArray nuiToMarkers(wvu_swarm_std_msgs::nuitrack_data _nui)
{
    visualization_msgs::MarkerArray ret;

    ret.markers.push_back(pointToMarker(1, _nui.leftWrist, 1, 0.5, 0, 1));
    ret.markers.push_back(pointToMarker(2, _nui.leftHand, 1, 0, 0, 1, _nui.leftClick ? 6 : 4));
    ret.markers.push_back(pointToMarker(3, _nui.rightWrist, 0, 0.5, 1, 1));
    ret.markers.push_back(pointToMarker(4, _nui.rightHand, 0, 0, 1, 1, _nui.rightClick ? 6 : 4));

    return ret;
}

int main(int argc, char** argv) {
    // ROS setup
    ros::init(argc, argv, "rolling_average");

    // Generates nodehandles, publishers, subscriber
    ros::NodeHandle n;
    ros::NodeHandle n_priv("~"); // private handle
    ros::Publisher pub, pub_vis;
    ros::Subscriber sub;
    pub = n.advertise<wvu_swarm_std_msgs::nuitrack_data>("rolling_average", 1000);
    pub_vis = n.advertise<visualization_msgs::MarkerArray>("smoothed_visualization", 1000);
    sub = n.subscribe("unfiltered", 10, &msgCallback);

    // Import parameters from launchfile
    int maxSize;
    n_priv.param<int>("max_values", maxSize, 15);

    // Create a list to keep track of values
    // TODO: parametrize max size
    std::list<wvu_swarm_std_msgs::nuitrack_data> memory;

    wvu_swarm_std_msgs::nuitrack_data currentState;

    while (ros::ok()) {
        currentState = *(ros::topic::waitForMessage<wvu_swarm_std_msgs::nuitrack_data>("unfiltered"));

        // Remove a value if queue is too large
        if (memory.size() > maxSize) memory.pop_back();

        // Add current value to memory
        memory.push_front(currentState);

        // Find sum of all current values
        wvu_swarm_std_msgs::nuitrack_data sum;
        for (wvu_swarm_std_msgs::nuitrack_data nui : memory) {
            sum = combineNuiData(sum, nui);
        }

        // Divide for mean
        wvu_swarm_std_msgs::nuitrack_data mean = divideNuiData(sum, (double) memory.size());

        // Publish
        pub.publish(mean);
        pub_vis.publish(nuiToMarkers(mean));
    }

    return 0;
}
