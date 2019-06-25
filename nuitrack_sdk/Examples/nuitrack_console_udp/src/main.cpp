#include <nuitrack/Nuitrack.h>

#include <iomanip>
#include <iostream>

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

using namespace tdv::nuitrack;

// Global variables for UDP
int sockfd;
struct sockaddr_in servaddr, cliaddr; // netinet/in.h objects

// Global variables for hand data
double x, y, z;

// Global variables for threads
std::thread server, tracker;
bool sigint_received = false;


void flagger(int sig)
{
    sigint_received = true;
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

    auto rightHand = userHands[0].rightHand;
    if (!rightHand)
    {
        // No right hand
        std::cout << "Right hand of the first user is not found" << std::endl;
        return;
    }

//    std::cout << std::fixed << std::setprecision(3);
//    std::cout << "Right hand position: "
//                 "x = " << rightHand->xReal << ", "
//                 "y = " << rightHand->yReal << ", "
//                 "z = " << rightHand->zReal << std::endl;
    x = rightHand->xReal;
    y = rightHand->yReal;
    z = rightHand->zReal;
}

void serverResponse(char rec)
{
    double send = 0.0;
    
    if(rec == 'x')
        send = x;
    else if(rec == 'y')
        send = y;
    else if(rec == 'z')
        send = z;
    
    // Send data
    if(sendto(sockfd, &send, sizeof(send), MSG_CONFIRM,
            (const struct sockaddr*)&cliaddr, sizeof(cliaddr)) >= 0)
    {
        std::cout << "Sent " << send << " to client." << std::endl;
    }
    else
    {
        std::cout << "Error in sending!" << std::endl;
    }
    
    // Reset client address
    memset(&cliaddr, 0, sizeof(cliaddr)); 
}

void serverLoop(void)
{
    std::cout << "Entered server loop." << std::endl;
    int recVal;
    fd_set rfds;

    
    while(!sigint_received)
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

void trackerLoop(HandTracker::Ptr handTracker)
{
    std::cout << "Entered tracker loop." << std::endl;
    while (!sigint_received)
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
    
    // Release Nuitrack
    std::cout << "Releasing Nuitrack!" << std::endl;
    try
    {
        Nuitrack::release();
    }
    catch (const Exception& e)
    {
        std::cerr << "Nuitrack release failed (ExceptionType: " << e.type() << ")" << std::endl;
        return;
    }
    
    std::cout << "Successfully released Nuitrack." << std::endl;
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

    // Connect onHandUpdate callback to receive hand tracking data
    handTracker->connectOnUpdate(onHandUpdate);

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
    server = std::thread(serverLoop);
    tracker = std::thread(trackerLoop, handTracker);

    // Join threads after exiting from sigint
    server.join();
    tracker.join();

    return 1;
}
