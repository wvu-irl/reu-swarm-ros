#include <iostream>

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
#define MAXLINE 1024

bool sigint_received = false;

void flagger(int sig)
{
    sigint_received = true;
}

int main(int argc, char** argv) {
    // Set up signal handler
    signal(SIGINT, flagger);
    
    int sockfd;
    char buffer[MAXLINE];
    char *send = "Client wants data!";
    struct sockaddr_in servaddr;
    
    // Attempt to create the socket file descriptor
    if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Socket creation failed!" << std::endl;
        return -1;
    }
    
    // Fill the data structures with server information
    servaddr.sin_family = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY; // Accept messages from any address
    servaddr.sin_port = htons(PORT); // Sets port number, converts byte order
    
    double test = 0.0;
    
    while(!sigint_received)
    {
        // Send to server
//        if(sendto(sockfd, (const char*)send, strlen(send), MSG_CONFIRM,
//                (const struct sockaddr*)&servaddr, sizeof(servaddr)) >= 0)
        if(sendto(sockfd, (const double*)send, sizeof(double), MSG_CONFIRM,
                (const struct sockaddr*)&servaddr, sizeof(servaddr)) >= 0)
        {
            std::cout << "Sent to server." << std::endl;
        }
        else
        {
            std::cout << "Error in sending!" << std::endl;
        }
        sleep(1);
    }
    
    close(sockfd);
    std::cout << "Closed socket." << std::endl;
    
    return 0;
}

