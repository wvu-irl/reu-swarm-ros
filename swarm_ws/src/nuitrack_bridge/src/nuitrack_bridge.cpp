#include <iostream>
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
    
    /* ROTATION MATRIX 
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
    
    /* ALTERNATE ROTATION MATRIX
     *     [(1-2y2-2z2)  2(xy-wz)    2(wy+xz)  ]
     * R = [ 2(xy+wz)   (1-2x2-2z2)  2(yz-wx)  ]
     *     [ 2(xz+wy)    2(wx+yz)   (1-2x2-2y2)]
     */
    
    tfMat[0][0] = 1-2*y*y-2*z*z;
    tfMat[1][0] = 2*(x*y+w*z);
    tfMat[2][0] = 2*(x*z+w*y);
    tfMat[0][1] = 2*(x*y-w*z);
    tfMat[1][1] = 1-2*x*x-2*z*z;
    tfMat[2][1] = 2*(w*x+y*z);
    tfMat[0][2] = 2*(w*y+x*z);
    tfMat[1][2] = 2*(y*z-w*x);
    tfMat[2][2] = 1-2*x*x-2*y*y;
    
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
    double xin, yin, zin, xout, yout, zout;
    xin = _xyz.x / 1000.0;
    yin = _xyz.y / 1000.0;
    zin = _xyz.z / 1000.0;
    
    /* MATRIX MULTIPLICATION
     *   [x] [x']
     * T*[y]=[y']
     *   [z] [z']
     *   [1] [1 ]
     */
    xout = tfMat[0][0]*xin + tfMat[0][1]*yin + tfMat[0][2]*zin + tfMat[0][3];
    yout = tfMat[1][0]*xin + tfMat[1][1]*yin + tfMat[1][2]*zin + tfMat[1][3];
    zout = tfMat[2][0]*xin + tfMat[2][1]*yin + tfMat[2][2]*zin + tfMat[2][3];
//    xout = xin;
//    yout = yin;
//    zout = zin;
    
    // Apply the transform
    _xyz.x = xout;
    _xyz.y = yout;
    _xyz.z = zout;
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

visualization_msgs::MarkerArray nuiToMarkers(nuiData *_nui)
{
    visualization_msgs::Marker lW, lH, rW, rH;
    
    lW.header.stamp = ros::Time();
    lW.header.frame_id = "world";
    lW.ns = "hand";
    lW.id = 1;
    lW.type = visualization_msgs::Marker::SPHERE;
    lW.pose.position.x = _nui->leftWrist.x;
    lW.pose.position.y = _nui->leftWrist.y;
    lW.pose.position.z = _nui->leftWrist.z;
    lW.pose.orientation.x = 0;
    lW.pose.orientation.y = 0;
    lW.pose.orientation.z = 0;
    lW.pose.orientation.w = 1;
    lW.scale.x = 0.04;
    lW.scale.y = 0.04;
    lW.scale.z = 0.04;
    lW.color.r = 1;
    lW.color.g = 0.5;
    lW.color.b = 0;
    lW.color.a = 1;
    
    lH.header.stamp = ros::Time();
    lH.header.frame_id = "world";
    lH.ns = "hand";
    lH.id = 2;
    lH.type = visualization_msgs::Marker::SPHERE;
    lH.pose.position.x = _nui->leftHand.x;
    lH.pose.position.y = _nui->leftHand.y;
    lH.pose.position.z = _nui->leftHand.z;
    lH.pose.orientation.x = 0;
    lH.pose.orientation.y = 0;
    lH.pose.orientation.z = 0;
    lH.pose.orientation.w = 1;
    lH.scale.x = 0.04;
    lH.scale.y = 0.04;
    lH.scale.z = 0.04;
    lH.color.r = 1;
    lH.color.g = 0;
    lH.color.b = 0;
    lH.color.a = 1;
    
    rW.header.stamp = ros::Time();
    rW.header.frame_id = "world";
    rW.ns = "hand";
    rW.id = 3;
    rW.type = visualization_msgs::Marker::SPHERE;
    rW.pose.position.x = _nui->rightWrist.x;
    rW.pose.position.y = _nui->rightWrist.y;
    rW.pose.position.z = _nui->rightWrist.z;
    rW.pose.orientation.x = 0;
    rW.pose.orientation.y = 0;
    rW.pose.orientation.z = 0;
    rW.pose.orientation.w = 1;
    rW.scale.x = 0.04;
    rW.scale.y = 0.04;
    rW.scale.z = 0.04;
    rW.color.r = 0;
    rW.color.g = 0.5;
    rW.color.b = 1;
    rW.color.a = 1;
    
    rH.header.stamp = ros::Time();
    rH.header.frame_id = "world";
    rH.ns = "hand";
    rH.id = 4;
    rH.type = visualization_msgs::Marker::SPHERE;
    rH.pose.position.x = _nui->rightHand.x;
    rH.pose.position.y = _nui->rightHand.y;
    rH.pose.position.z = _nui->rightHand.z;
    rH.pose.orientation.x = 0;
    rH.pose.orientation.y = 0;
    rH.pose.orientation.z = 0;
    rH.pose.orientation.w = 1;
    rH.scale.x = 0.04;
    rH.scale.y = 0.04;
    rH.scale.z = 0.04;
    rH.color.r = 0;
    rH.color.g = 0;
    rH.color.b = 1;
    rH.color.a = 1;
    
    visualization_msgs::MarkerArray ret;
    ret.markers.push_back(lW);
    ret.markers.push_back(lH);
    ret.markers.push_back(rW);
    ret.markers.push_back(rH);
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
    ros::Publisher pub_v;
    pub = n.advertise<geometry_msgs::Point>("hand", 1000);
    pub_v = n.advertise<visualization_msgs::MarkerArray>("hand_vis", 1000);
    
    tf::TransformListener listener;
    
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
    
    nuiData nui;
    
    while(!g_sigint_received && ros::ok())
    {
        /* Send to server */
//        std::cout << "Char to send: ";
//        std::cin >> send;
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
                
                nuiData rec;
                n = recvfrom(sockfd, &rec, sizeof(rec), MSG_WAITALL,
                        (struct sockaddr*)&cliaddr, (socklen_t*)&len);
                
                // Transform the data
                nuiTf(rec);
                
                nui = rec;
                
                break;
            }
        }
        /* End wait for response */
        
        pub_v.publish(nuiToMarkers(&nui));
        
        ros::spinOnce();
        rate.sleep();
    }
    
    close(sockfd);
    std::cout << "Closed socket." << std::endl;
    
    return 0;
}