#include <iostream>
#include <stdio.h>
#include <nuitrack_bridge/nuitrack_data.h>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>

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

// Matrix for transformation from Kinect to global frame
double tfMat[4][4];

bool g_sigint_received = false;

void flagger(int sig)
{
    g_sigint_received = true;
}

// Function to set up transformation matrix
void initTf(void)
{
    double w, x, y, z, offsetX, offsetY, offsetZ;
    w = KINECT_QUAT_W;
    x = KINECT_QUAT_X;
    y = KINECT_QUAT_Y;
    z = KINECT_QUAT_Z;
    offsetX = KINECT_TRAN_X;
    offsetY = KINECT_TRAN_Y;
    offsetZ = KINECT_TRAN_Z;
    
    /* ROTATION MATRIX, HOMOGENOUS QUATERNION
     *     [(w2+x2-y2-z2)   2(xy-wz)      2(wy+xz)   ]
     * R = [  2(xy+wz)    (w2-x2+y2-z2)   2(yz-wx)   ]
     *     [  2(xz+wy)      2(wx+yz)    (w2-x2-y2+z2)]
     */
    
//    tfMat[0][0] = w*w+x*x-y*y-z*z;
//    tfMat[1][0] = 2*(x*y+w*z);
//    tfMat[2][0] = 2*(x*z+w*y);
//    tfMat[0][1] = 2*(x*y-w*z);
//    tfMat[1][1] = w*w-x*x+y*y-z*z;
//    tfMat[2][1] = 2*(w*x+y*z);
//    tfMat[0][2] = 2*(w*y+x*z);
//    tfMat[1][2] = 2*(y*z-w*x);
//    tfMat[2][2] = w*w-x*x-y*y+z*z;
    
    /* ROTATION MATRIX, NONHOMOGENOUS QUATERNION
     *     [(1-2sy2-2sz2)  2s(xy-wz)    2(swy+xz)  ]
     * R = [ 2s(xy+wz)   (1-2sx2-2sz2)  2s(yz-wx)  ]
     *     [ 2s(xz+wy)    2s(wx+yz)   (1-2sx2-2sy2)]
     */
    
    // Variable needed for normalization (our matrix may be slightly off)
    double s = 1.0 / ((w*w + x*x + y*y + z*z)*(w*w + x*x + y*y + z*z));
    
    tfMat[0][0] = 1-2*s*y*y-2*s*z*z;
    tfMat[1][0] = 2*s*(x*y+w*z);
    tfMat[2][0] = 2*s*(x*z+w*y);
    tfMat[0][1] = 2*s*(x*y-w*z);
    tfMat[1][1] = 1-2*s*x*x-2*s*z*z;
    tfMat[2][1] = 2*s*(w*x+y*z);
    tfMat[0][2] = 2*s*(w*y+x*z);
    tfMat[1][2] = 2*s*(y*z-w*x);
    tfMat[2][2] = 1-2*s*x*x-2*s*y*y;
    
    // Fill in translation
    tfMat[0][3] = offsetX;
    tfMat[1][3] = offsetY;
    tfMat[2][3] = offsetZ;
    
    // Fill in constants
    tfMat[3][0] = 0;
    tfMat[3][1] = 0;
    tfMat[3][2] = 0;
    tfMat[3][3] = 1;
}

// Function to transform a single xyz
void xyzTf(xyz &_xyz)
{
    // Get the transform
    tf::TransformListener tfLis;
    
    // Create a geometry message point because tf needs it
    geometry_msgs::PointStamped pt;
    pt.header.frame_id = "kinect";
    pt.header.stamp = ros::Time();
    pt.point.x = _xyz.x / 1000; // mm to m
    pt.point.y = _xyz.y / 1000;
    pt.point.z = _xyz.z / 1000;
    
    try {
        // Another placeholder point to transform into
        geometry_msgs::PointStamped ptTf;
        
        // Transform it into world frame
        tfLis.transformPoint("world", pt, ptTf);
        
        // Apply to input
        _xyz.x = ptTf.point.x;
        _xyz.y = ptTf.point.y;
        _xyz.z = ptTf.point.z;
    }
    catch (tf::TransformException &ex) {
        ROS_ERROR("Transform error! %s", ex.what());
    }
}


// Function to transform an nuiData struct from Kinect to global frame
void nuiTf(nuiData &_nui)
{
    // Transform each xyz point
    if(_nui.leftFound) {
        xyzTf(_nui.leftHand);
        xyzTf(_nui.leftWrist);
    }
    if(_nui.rightFound) {
        xyzTf(_nui.rightHand);
        xyzTf(_nui.rightWrist);
    }
}

visualization_msgs::Marker xyzToMarker(int _id, xyz *_xyz, double _r, double _g, double _b)
{
    visualization_msgs::Marker ret;
    
    ret.header.stamp = ros::Time();
    ret.header.frame_id = "world";
    ret.ns = "hand";
    ret.id = _id;
    ret.type = visualization_msgs::Marker::SPHERE;
    ret.pose.position.x = _xyz->x;
    ret.pose.position.y = _xyz->y;
    ret.pose.position.z = _xyz->z;
    ret.pose.orientation.x = 0;
    ret.pose.orientation.y = 0;
    ret.pose.orientation.z = 0;
    ret.pose.orientation.w = 1;
    ret.scale.x = 0.04;
    ret.scale.y = 0.04;
    ret.scale.z = 0.04;
    ret.color.r = _r;
    ret.color.g = _g;
    ret.color.b = _b;
    ret.color.a = 1;
    
    return ret;
}

visualization_msgs::MarkerArray nuiToMarkers(nuiData *_nui)
{
    visualization_msgs::MarkerArray ret;
    
    ret.markers.push_back(xyzToMarker(1, &(_nui->leftWrist), 1, 0.5, 0));
    ret.markers.push_back(xyzToMarker(2, &(_nui->leftHand), 1, 0, 0));
    ret.markers.push_back(xyzToMarker(3, &(_nui->rightWrist), 0, 0.5, 1));
    ret.markers.push_back(xyzToMarker(4, &(_nui->rightHand), 0, 0, 1));
    
    return ret;
}

int main(int argc, char** argv) {
    // Set up transformation matrix
    initTf();
    
    // ROS setup
    ros::init(argc, argv, "nuitrack_bridge");
    
    // Generates nodehandles, publisher
    ros::NodeHandle n;
    ros::NodeHandle n_priv("~"); // private handle
    ros::Rate rate(15);
    ros::Publisher pub;
    pub = n.advertise<visualization_msgs::MarkerArray>("nuitrack_bridge", 1000);
    
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
        send = 'a';
        if(sendto(sockfd, &send, sizeof(send), MSG_CONFIRM,
                (const struct sockaddr*)&servaddr, sizeof(servaddr)) >= 0)
        {
            std::cout << "Sent " << send << " to server." << std::endl;
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
                std::cout << "Socket is available." << std::endl;
                
                int len, n;
                
                nuiData rec = nuiData();
                n = recvfrom(sockfd, &rec, sizeof(rec), MSG_WAITALL,
                        (struct sockaddr*)&cliaddr, (socklen_t*)&len);
                
                // Transform the data
                nui = rec;
    printf("LH: %02.3f, %02.3f, %02.3f\n\r", nui.leftHand.x, nui.leftHand.y, nui.leftHand.z);
    printf("LW: %02.3f, %02.3f, %02.3f\n\r", nui.leftWrist.x, nui.leftWrist.y, nui.leftWrist.z);
    printf("RH: %02.3f, %02.3f, %02.3f\n\r", nui.rightHand.x, nui.rightHand.y, nui.rightHand.z);
    printf("RW: %02.3f, %02.3f, %02.3f\n\r", nui.rightWrist.x, nui.rightWrist.y, nui.rightWrist.z);
                nuiTf(nui);
                
                break;
            }
        }
        /* End wait for response */
        
        visualization_msgs::MarkerArray vis = nuiToMarkers(&nui);
        pub.publish(vis);
        
        ros::spinOnce();
        rate.sleep();
    }
    
    close(sockfd);
    std::cout << "Closed socket." << std::endl;
    
    return 0;
}