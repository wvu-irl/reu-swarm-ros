#include <nuitrack/Nuitrack.h>

#include <nuitrack_bridge/nuitrack_data.h>

#include <iomanip>
#include <iostream>
#include <stdio.h>

// UDP includes
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/time.h>
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 

// Threading-based includes
#include <thread>
#include <signal.h>
  
#define DEBUG 1
#define PORT 8080 

// When 1, use handTracker for hand location. When 0,
//   use skeletonTracker's hand joint for hand location.
#define USEHAND 1

using namespace tdv::nuitrack;

// Global variables for UDP
int sockfd;
struct sockaddr_in servaddr, cliaddr; // netinet/in.h objects

// Global variable for hand data
nuiData nui;

// Global variables for threads
std::thread g_server, g_hTracker, g_sTracker;
bool g_sigint_received = false;


void flagger(int sig)
{
    g_sigint_received = true;
}

// Callback for the hand data update event
void onHandUpdate(HandTrackerData::Ptr handData)
{
    if (!handData)
    {
        // No hand data
        std::cout << "No hand data" << std::endl;
        return;
    }

    auto userHands = handData->getUsersHands();
    if (userHands.empty())
    {
        // No user hands
        return;
    }

#if USEHAND
    auto rightHand = userHands[0].rightHand;
    if (!rightHand)
    {
        // No right hand
        std::cout << "Right hand of the first user is not found" << std::endl;
        return;
    }
    else
    {
        // handTracker left/right is opposite of skelTracker
        nui.leftHand.x = rightHand->xReal;
        nui.leftHand.y = rightHand->yReal;
        nui.leftHand.z = rightHand->zReal;
    }
    
    auto leftHand = userHands[0].leftHand;
    if (!leftHand)
    {
        // No right hand
        std::cout << "Left hand of the first user is not found" << std::endl;
        return;
    }
    else
    {
        // handTracker left/right is opposite of skelTracker
        nui.rightHand.x = leftHand->xReal;
        nui.rightHand.y = leftHand->yReal;
        nui.rightHand.z = leftHand->zReal;
    }
#endif
}

void onSkelUpdate(SkeletonData::Ptr skelData)
{
    if (!skelData)
    {
        // No data received
        std::cout << "No skeleton data" << std::endl;
        return;
    }
    
    auto skeletons = skelData->getSkeletons();
    if (skeletons.empty())
    {
        // No skeletons somehow
        return;
    }
    
    Joint leftWrist, leftHand, rightWrist, rightHand;
    nui.leftFound = true;
    nui.rightFound = true; //assume both found
    
    try {
        leftWrist = skeletons.at(0).joints.at(JOINT_LEFT_WRIST);
    }
    catch (const Exception& e) {
        // No right hand
        std::cout << "Left wrist of the first user is not found" << std::endl;
        nui.leftFound = false;
    }
    
    try {
        rightWrist = skeletons.at(0).joints.at(JOINT_RIGHT_WRIST);
    }
    catch (const Exception& e) {
        // No right hand
        std::cout << "Right wrist of the first user is not found" << std::endl;
        nui.rightFound = false;
    }
    
#if !USEHAND
    try {
        leftHand = skeletons.at(0).joints.at(JOINT_LEFT_HAND);
    }
    catch (const Exception& e) {
        // No right hand
        std::cout << "Left hand of the first user is not found" << std::endl;
        nui.leftFound = false;
    }
    
    try {
        rightHand = skeletons.at(0).joints.at(JOINT_RIGHT_HAND);
    }
    catch (const Exception& e) {
        // No right hand
        std::cout << "Right hand of the first user is not found" << std::endl;
        nui.rightFound = false;
    }
#endif
    
    //Pull data into struct
    if(nui.leftFound)
    {
        nui.leftWrist.x = leftWrist.real.x;
        nui.leftWrist.y = leftWrist.real.y;
        nui.leftWrist.z = leftWrist.real.z;
#if !USEHAND
        nui.leftHand.x = leftHand.real.x;
        nui.leftHand.y = leftHand.real.y;
        nui.leftHand.z = leftHand.real.z;
#endif
    }
    else
    {
        nui.leftWrist = xyz();
#if !USEHAND
        nui.leftHand = xyz();
#endif
    }
    if(nui.rightFound)
    {
        nui.rightWrist.x = rightWrist.real.x;
        nui.rightWrist.y = rightWrist.real.y;
        nui.rightWrist.z = rightWrist.real.z;
#if !USEHAND
        nui.rightHand.x = rightHand.real.x;
        nui.rightHand.y = rightHand.real.y;
        nui.rightHand.z = rightHand.real.z;
#endif
    }
    else
    {
        nui.rightWrist = xyz();
#if !USEHAND
        nui.rightHand = xyz();
#endif
    }
}

void serverResponse(char rec)
{
    nuiData send;
    
    // Send data
    send = nui;
    if(sendto(sockfd, (void*)&send, sizeof(send), MSG_CONFIRM,
            (const struct sockaddr*)&cliaddr, sizeof(cliaddr)) >= 0)
    {
        std::cout << "Sent " /*<< send*/ << "to client." << std::endl;
    }
    else
    {
        std::cout << "Error in sending!" << std::endl;
    }
    
    printf("LH: %02.3f, %02.3f, %02.3f\n\r", nui.leftHand.x, nui.leftHand.y, nui.leftHand.z);
    printf("LW: %02.3f, %02.3f, %02.3f\n\r", nui.leftWrist.x, nui.leftWrist.y, nui.leftWrist.z);
    printf("RH: %02.3f, %02.3f, %02.3f\n\r", nui.rightHand.x, nui.rightHand.y, nui.rightHand.z);
    printf("RW: %02.3f, %02.3f, %02.3f\n\r", nui.rightWrist.x, nui.rightWrist.y, nui.rightWrist.z);
    
    // Reset client address
    memset(&cliaddr, 0, sizeof(cliaddr)); 
}

void serverLoop(void)
{
    std::cout << "Entered server loop." << std::endl;
    int recVal;
    fd_set rfds;

    
    while(!g_sigint_received)
    {
        // Set up socket to watch in select()
        FD_ZERO(&rfds);
        FD_SET(sockfd, &rfds);
        
        // Set timeout to 10 ms
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;//10000;
        
        // Check if socket is available (non-blocking) for up to timeout seconds
        recVal = select(sockfd + 2, &rfds, NULL, NULL, &timeout);
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
                char rec;
                
                // Read into rec
                n = recvfrom(sockfd, &rec, sizeof(rec), MSG_WAITALL,
                        (struct sockaddr*)&cliaddr, (socklen_t*)&len);

                std::cout << "Client sent " << (uint8_t)rec << std::endl;
                
                // Handle this request
                serverResponse(rec);
                
                break;
            }
        }
    }
    
    std::cout << "Closing socket!" << std::endl;
    close(sockfd);
    
    std::cout << "Successfully closed server." << std::endl;
}

void handTrackerLoop(HandTracker::Ptr handTracker)
{
    std::cout << "Entered hand tracker loop." << std::endl;
    while (!g_sigint_received)
    {
        try
        {
            // Wait for new hand tracking data
            Nuitrack::waitUpdate(handTracker);
        }
        catch (LicenseNotAcquiredException& e)
        {
            std::cerr << "LicenseNotAcquired exception (ExceptionType: " << e.type() << ")" << std::endl;
            break;
        }
        catch (const Exception& e)
        {
            std::cerr << "Nuitrack update failed (ExceptionType: " << e.type() << ")" << std::endl;
        }
    }
}

void skelTrackerLoop(SkeletonTracker::Ptr skelTracker)
{
    std::cout << "Entered skeleton tracker loop." << std::endl;
    while (!g_sigint_received)
    {
        try
        {
            // Wait for new hand tracking data
            Nuitrack::waitUpdate(skelTracker);
        }
        catch (LicenseNotAcquiredException& e)
        {
            std::cerr << "LicenseNotAcquired exception (ExceptionType: " << e.type() << ")" << std::endl;
            break;
        }
        catch (const Exception& e)
        {
            std::cerr << "Nuitrack update failed (ExceptionType: " << e.type() << ")" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    /* SERVER SETUP */
    // Attempt to create the socket file descriptor
    if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Socket creation failed!" << std::endl;
        return -1;
    }
    std::cout << "Socket successfully created." << std::endl;
    
    // Zero out addresses
    memset(&servaddr, 0, sizeof(servaddr)); 
    memset(&cliaddr, 0, sizeof(cliaddr)); 
    
    // Fill the data structures with server information
    servaddr.sin_family = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY; // Accept messages from any address
    servaddr.sin_port = htons(PORT); // Sets port number, converts byte order
    
    // Attempt to bind the socket
    if(bind(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        std::cerr << "Error binding to port " << PORT << "!" << std::endl;
        return -2;
    }
    std::cout << "Successfully bound to port " << PORT << "." << std::endl;
    
    /* NUITRACK SETUP */
    try
    {
        Nuitrack::init("");
    }
    catch (const Exception& e)
    {
        std::cerr << "Can not initialize Nuitrack (ExceptionType: " << e.type() << ")" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Create HandTracker module, other required modules will be
    // created automatically
    auto handTracker = HandTracker::create();
    auto skelTracker = SkeletonTracker::create();

    // Connect onHandUpdate callback to receive hand tracking data
    handTracker->connectOnUpdate(onHandUpdate);
    skelTracker->connectOnUpdate(onSkelUpdate);

    // Start Nuitrack
    try
    {
        Nuitrack::run();
    }
    catch (const Exception& e)
    {
        std::cerr << "Can not start Nuitrack (ExceptionType: " << e.type() << ")" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Set up signal handler
    signal(SIGINT, flagger);
    
    // Start a thread for the server handler and the nuitrack handler
    g_server = std::thread(serverLoop);
    g_hTracker = std::thread(handTrackerLoop, handTracker);
    g_sTracker = std::thread(skelTrackerLoop, skelTracker);
    
    // Join threads after exiting from sigint
    g_server.join();
    g_hTracker.join();
    g_sTracker.join();
    
    // Release Nuitrack
    std::cout << "Releasing Nuitrack!" << std::endl;
    try
    {
        Nuitrack::release();
    }
    catch (const Exception& e)
    {
        std::cerr << "Nuitrack release failed (ExceptionType: " << e.type() << ")" << std::endl;
        return -1;
    }
    
    std::cout << "Successfully released Nuitrack." << std::endl;

    return 1;
}
