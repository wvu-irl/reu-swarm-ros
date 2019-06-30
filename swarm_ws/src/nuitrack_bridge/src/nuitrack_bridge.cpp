#include <iostream>
#include <nuitrack_bridge/nuitrack_data.h>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

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
    
    /* ROTATION MATRIX
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
    xin = _xyz.x;
    yin = _xyz.y;
    zin = _xyz.z;
    
    /* MATRIX MULTIPLICATION
     *   [x] [x']
     * T*[y]=[y']
     *   [z] [z']
     *   [1] [1 ]
     */
    xout = tfMat[0][0]*xin + tfMat[0][1]*yin + tfMat[0][2]*zin + tfMat[0][3];
    yout = tfMat[1][0]*xin + tfMat[1][1]*yin + tfMat[1][2]*zin + tfMat[1][3];
    zout = tfMat[2][0]*xin + tfMat[2][1]*yin + tfMat[2][2]*zin + tfMat[2][3];
    
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
        xyzTf(_nui.leftWrist);
    }
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
    
    double x = 0.0, y = 0.0, z = 0.0;
    double a = 0.0, b = 0.0, c = 0.0;
    
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
                
                rec.rightHand.x /= 100;
                rec.rightHand.y /= 100;
                rec.rightHand.z /= 100;
                
                a = rec.rightHand.x;
                b = rec.rightHand.y;
                c = rec.rightHand.z;
                
                // Transform the data
                nuiTf(rec);
                
                x = rec.rightHand.x;
                y = rec.rightHand.y;
                z = rec.rightHand.z;
                
                break;
            }
        }
        /* End wait for response */
        
        // Create object to publish
        geometry_msgs::Point output;
        output.x = x;
        output.y = y;
        output.z = z;
        
        pub.publish(output);
        
        // Build some Markers for visualization
        visualization_msgs::Marker rawMark, tfMark;
        
        rawMark.header.stamp = ros::Time();
        rawMark.header.frame_id = "map";
        rawMark.ns = "hand";
        rawMark.id = 1;
        rawMark.type = visualization_msgs::Marker::SPHERE;
        rawMark.pose.position.x = a;
        rawMark.pose.position.y = b;
        rawMark.pose.position.z = c;
        rawMark.pose.orientation.x = 0;
        rawMark.pose.orientation.y = 0;
        rawMark.pose.orientation.z = 0;
        rawMark.pose.orientation.w = 1;
        rawMark.scale.x = 1; //0.04;
        rawMark.scale.y = 1; //0.04;
        rawMark.scale.z = 1; //0.04;
        rawMark.color.r = 1;
        rawMark.color.g = 0;
        rawMark.color.b = 0;
        rawMark.color.a = 1;
        
        tfMark.header.stamp = ros::Time();
        tfMark.header.frame_id = "map";
        tfMark.ns = "hand";
        tfMark.id = 2;
        tfMark.type = visualization_msgs::Marker::SPHERE;
        tfMark.pose.position.x = x;
        tfMark.pose.position.y = y;
        tfMark.pose.position.z = z;
        tfMark.pose.orientation.x = 0;
        tfMark.pose.orientation.y = 0;
        tfMark.pose.orientation.z = 0;
        tfMark.pose.orientation.w = 1;
        tfMark.scale.x = 1; //0.04;
        tfMark.scale.y = 1; //0.04;
        tfMark.scale.z = 1; //0.04;
        tfMark.color.r = 0;
        tfMark.color.g = 1;
        tfMark.color.b = 0;
        tfMark.color.a = 1;
        
        visualization_msgs::MarkerArray visOutput;
        visOutput.markers = {rawMark, tfMark};
        pub_v.publish(visOutput);
        
        ros::spinOnce();
        rate.sleep();
    }
    
    close(sockfd);
    std::cout << "Closed socket." << std::endl;
    
    return 0;
}

