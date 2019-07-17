/*
 * NUITRACK BRIDGE
 * This code is meant to act as a bridge with NuiTrack, allowing integration
 * of a skeleton and human gestures to our ROS network. Because NuiTrack and ROS
 * refuse to compile together, this program acts as the ROS side.
 * This program looks for a NuiTrack data on a local UDP port which must be sent
 * from a complementary server, found at reu-swarm-ros/nuitrack_sdk/Examples/nuitrack_console_udp
 * Thus, that must be running for this node to function properly.
 */

#include <iostream>
#include <stdio.h>
#include <nuitrack_bridge/nuitrack_data.h>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <wvu_swarm_std_msgs/nuitrack_data.h>

// UDP includes
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <signal.h>

#define PORT 8080

bool g_sigint_received = false;

void flagger(int sig)
{
    g_sigint_received = true;
}

// Function to transform a single xyz
void xyzTf(xyz &_xyz, tf::TransformListener &tfLis)
{
    // Create a geometry message point because tf needs it
    geometry_msgs::PointStamped pt;
    pt.header.frame_id = "kinect";
    pt.header.stamp = ros::Time();
    pt.point.x = _xyz.x / 1000; // mm to meters
    pt.point.y = _xyz.y / 1000;
    pt.point.z = _xyz.z / 1000;

    try {
        // Another placeholder point to transform into
        geometry_msgs::PointStamped ptTf;

        // Find transform from kinect to world
        tfLis.waitForTransform("kinect", "world", ros::Time(), ros::Duration(0.01));
        tfLis.transformPoint("world", pt, ptTf); // Transform pt into world, store in ptTf

        // Apply to input
        _xyz.x = ptTf.point.x * 100; // meters to cm
        _xyz.y = ptTf.point.y * 100;
        _xyz.z = ptTf.point.z * 100;
    }
    catch (tf::TransformException &ex) {
        ROS_ERROR("Transform error! %s", ex.what());
    }
}

// Function to transform an nuiData struct from Kinect to global frame
void nuiTf(nuiData &_nui, tf::TransformListener &tfLis)
{
    // Transform each xyz point
    if(_nui.leftFound) {
        xyzTf(_nui.leftHand, tfLis);
        xyzTf(_nui.leftWrist, tfLis);
    }
    if(_nui.rightFound) {
        xyzTf(_nui.rightHand, tfLis);
        xyzTf(_nui.rightWrist, tfLis);
    }
}

// Generates a visualizable Marker from an xyz struct
visualization_msgs::Marker xyzToMarker(int _id, xyz *_xyz, double _r, double _g, double _b, double _a = 1.0, double _size = 4)
{
    visualization_msgs::Marker ret;

    ret.header.stamp = ros::Time(); // Metadata
    ret.header.frame_id = "world";
    ret.ns = "nui";
    ret.id = _id; // Unique identifier
    ret.type = visualization_msgs::Marker::SPHERE;
    ret.pose.position.x = _xyz->x; // Measured in meters
    ret.pose.position.y = _xyz->y;
    ret.pose.position.z = _xyz->z;
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

// Generates a simple Point message from an xyz struct
geometry_msgs::Point xyzToPoint(xyz *_xyz)
{
    geometry_msgs::Point ret;

    ret.x = _xyz->x;
    ret.y = _xyz->y;
    ret.z = _xyz->z;

    return ret;
}

// Generates an array to convert an entire nuiData struct into visualizable form
visualization_msgs::MarkerArray nuiToMarkers(nuiData *_nui)
{
    visualization_msgs::MarkerArray ret;

    // Use confidence values as alpha-- less certain joints will be more transparent
    double leftHandAlpha = 0.5; //_nui->confLH;
    double leftWristAlpha = 0.5; //_nui->confLW;
    double rightHandAlpha = 0.5; //_nui->confRH;
    double rightWristAlpha = 0.5; //_nui->confRW;

    ret.markers.push_back(xyzToMarker(1, &(_nui->leftWrist), 1, 0.5, 0, leftWristAlpha));
    ret.markers.push_back(xyzToMarker(2, &(_nui->leftHand), 1, 0, 0, leftHandAlpha, _nui->leftClick ? 6 : 4));
    ret.markers.push_back(xyzToMarker(3, &(_nui->rightWrist), 0, 0.5, 1, rightWristAlpha));
    ret.markers.push_back(xyzToMarker(4, &(_nui->rightHand), 0, 0, 1, rightHandAlpha, _nui->rightClick ? 6 : 4));

    return ret;
}

// Generates an nuitrack_data ROS message from an nuiData struct
wvu_swarm_std_msgs::nuitrack_data nuiToRos(nuiData *_nui)
{
    wvu_swarm_std_msgs::nuitrack_data ret;

    ret.leftWrist = xyzToPoint(&(_nui->leftWrist));
    ret.leftHand = xyzToPoint(&(_nui->leftHand));
    ret.rightWrist = xyzToPoint(&(_nui->rightWrist));
    ret.rightHand = xyzToPoint(&(_nui->rightHand));
    ret.gestureData = (char)(_nui->gestureData);
    ret.leftClick = _nui->leftClick;
    ret.rightClick = _nui->rightClick;

    return ret;
}

int main(int argc, char** argv) {
    // ROS setup
    ros::init(argc, argv, "nuitrack_bridge");

    // Generates nodehandles, publisher
    ros::NodeHandle n;
    ros::NodeHandle n_priv("~"); // private handle
    ros::Rate rate(50);
    ros::Publisher pubVis, pubNui;
    pubVis = n.advertise<visualization_msgs::MarkerArray>("visualization", 1000);
    pubNui = n.advertise<wvu_swarm_std_msgs::nuitrack_data>("unfiltered", 1000);

    // Transform listener
    tf::TransformListener tfLis;

    // Set up signal handler
    signal(SIGINT, flagger);

    /* Server setup */
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;

    // Attempt to create the socket file descriptor
    if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Socket creation failed!" << std::endl;
        return -1;
    }

    // Fill the data structures with server information
    servaddr.sin_family = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY; // Accept messages from any address
    servaddr.sin_port = htons(PORT); // Sets port number, converts byte order
    /* End server setup */

    char send = '\0';

    nuiData nui = nuiData();

    while(!g_sigint_received && ros::ok())
    {
        /* Send to server */
        send = 'a'; // For now, can send anything and server will respond.
        if(sendto(sockfd, &send, sizeof(send), MSG_CONFIRM,
                (const struct sockaddr*)&servaddr, sizeof(servaddr)) >= 0)
        {
//            std::cout << "Sent " << send << " to server." << std::endl;
        }
        else
        {
            std::cout << "Error in sending!" << std::endl;
        }
        /* End send to server */

        /* Wait for response */
        // Set up socket to watch in select()
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sockfd, &rfds);

        // Set timeout to 10 ms
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;//10000;

        // Check if socket is available (non-blocking) for up to timeout seconds
        int recVal = select(sockfd + 2, &rfds, NULL, NULL, &timeout);
        switch(recVal)
        {
            case(0):
            {
                //Timeout
                std::cout << "Timeout on socket." << std::endl;
                break;
            }
            case(-1):
            {
                //Error
                std::cout << "Select returned error!" << std::endl;
                break;
            }
            default:
            {
                // Data is available!
//                std::cout << "Socket is available." << std::endl;

                int len, n;
                nuiData rec = nuiData();

                // Data was found, receive it from the socket
                n = recvfrom(sockfd, &rec, sizeof(rec), MSG_WAITALL,
                        (struct sockaddr*)&cliaddr, (socklen_t*)&len);

                // Store in global
                nui = rec;

                // Transform global
                nuiTf(nui, tfLis);

                break;
            }
        }
        /* End wait for response */

        // Create and publish visualizable message for rviz
        visualization_msgs::MarkerArray visOut = nuiToMarkers(&nui);
        pubVis.publish(visOut);

        // Create and publish nutrack_data so other nodes can interpret
        wvu_swarm_std_msgs::nuitrack_data nuiOut = nuiToRos(&nui);
        pubNui.publish(nuiOut);

        ros::spinOnce();
        rate.sleep();
    }

    close(sockfd);
    std::cout << "Closed socket." << std::endl;

    return 0;
}
