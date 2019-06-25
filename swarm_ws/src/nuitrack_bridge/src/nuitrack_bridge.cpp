#include <iostream>
#include <nuitrack_bridge/nuitrack_data.h>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Point.h>

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

int main(int argc, char** argv) {
    // ROS setup
    ros::init(argc, argv, "nuitrack_bridge");
    
    // Generates nodehandles, publisher
    ros::NodeHandle n;
    ros::NodeHandle n_priv("~"); // private handle
    ros::Publisher pub;
    pub = n.advertise<geometry_msgs::Point>("hand", 1000);
    
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
    
    while(!g_sigint_received && ros::ok())
    {
        /* Send to server */
        std::cout << "Char to send: ";
        std::cin >> send;
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
                //double rec;
                
                // Read into rec
//                n = recvfrom(sockfd, &rec, sizeof(rec), MSG_WAITALL,
//                        (struct sockaddr*)&cliaddr, (socklen_t*)&len);
//
//                std::cout << "Client sent " << rec << std::endl;
//                
//                // Handle this request
//                if(send == 'x')
//                    x = rec;
//                else if(send == 'y')
//                    y = rec;
//                else if(send == 'z')
//                    z = rec;
                
                nuiData rec;
                n = recvfrom(sockfd, &rec, sizeof(rec), MSG_WAITALL,
                        (struct sockaddr*)&cliaddr, (socklen_t*)&len);
                x = rec.leftWristX;
                y = rec.leftWristY;
                z = rec.leftWristZ;
                
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
        ros::spinOnce();
    }
    
    close(sockfd);
    std::cout << "Closed socket." << std::endl;
    
    return 0;
}

