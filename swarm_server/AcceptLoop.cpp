/**
 * 
 * Server Accept Loop
 * 
 * Author: Henry Vos
 * 
 * Purpose:
 *      This section is to accept connections from all the robots and the ViCon system.
 *      The clients will each get their own process.
 * 
 */

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>

#include "ViCon.hpp"
#include "Robot.hpp"


// Setting global variables
int server_port = 4321; // port number

// datastructure that is used to send/recieve commands
typedef struct
{
    char *str;
} command;

char *toLowerCase(char *str)
{
    size_t leng = strlen(str);

    for (size_t i = 0; i < leng; i++)
    {
        if (str[i] >= 'A' && str[i] <= 'Z')
        {
            str[i] += 'a' - 'A';
        }
    }
    return str;
}

/**
 *  Function that proccesses all the commands from each indivudial client
 *  
 *  Split for robot and ViCon
 * 
 */
void proccessCommand(char *str)
{
    toLowerCase(str);
    // universal commands
    // register
    if (strstr("register", str) == str) // string starts with "register"
    {
        // vicon and bot are acceptable
        char type[16] = {'\0'};
        sscanf(str, "register %s", type); // retreiving type string from command

        // checking type
        if (strcmp(type, "vicon") == 0)
        {
        }
        else if (strcmp(type, "bot") == 0)
        {
        }
        else if (strcmp(type, "user"))
        {
        }
        else
        {
            // undefined registration
            printf("Registration error for type not recognised:%s\n", type);
        }
    }

    // echo
}

int main()
{
    // starting socket
    int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_descriptor == -1)
    {
        puts("Error getting socket.");
        exit(1);
    }

    // Change receive timeout to 30 seconds.
    struct timeval timeout;
    timeout.tv_sec = 30;
    setsockopt(socket_descriptor, SOL_SOCKET, SO_RCVTIMEO, (struct timeval *)&timeout, sizeof(struct timeval));

    // Instantiate struct to store socket settings.
    struct sockaddr_in socket_address;
    memset(&socket_address, 0, sizeof(socket_address));
    socket_address.sin_family = AF_INET;
    socket_address.sin_addr.s_addr = INADDR_ANY;
    socket_address.sin_port = htons(server_port);

    // Bind to the socket
    if (bind(socket_descriptor, (struct sockaddr *)&socket_address, sizeof(socket_address)) == -1)
    {
        puts("Error binding to socket.");
        exit(1);
    }

    // Set socket to listen for connections
    if (listen(socket_descriptor, 3) == -1)
    {
        puts("Error listening for connections.");
        exit(1);
    }

    pid_t currpid = getpid();
    int connection_descriptor;
    struct sockaddr connection_addr;
    int numConns = 0;
    // Loop to handle connections on the socket
    while (1)
    {
        if (currpid != 0)
        {
            // Specify struct to store information about accepted connection
            socklen_t connection_addr_size = sizeof(struct sockaddr);

            // Accept connection
            connection_descriptor = accept(socket_descriptor, &connection_addr, &connection_addr_size);
            if (connection_descriptor == -1)
            {
                puts("Error accepting connection.");
                continue;
            }
            else
            {
                numConns++;
                currpid = fork();                  // creating a client handling process
            }
        }
        else // in client handler
        {
            // Get char representation of client IP address
            char *client_ip = inet_ntoa(((struct sockaddr_in *)&connection_addr)->sin_addr);

            // Read data from connection into buffer. Continue to loop while message size is >0.
            int message_size = 0;
            char commandBuffer[64] = {'\0'};
            command buffer = {commandBuffer};

            int done = 0;
            int count = 0;

            //reading the message
            while ((message_size = read(connection_descriptor, &buffer, sizeof(buffer))) > 0)
            {
                proccessCommand(buffer.str); // sending message to somewhere else to make code more readable
            }

            // Display if there was an error
            if (message_size == -1)
            {
                puts("Error receiving message.");
                exit(1);
            }

            if (done)
                break;
        }
    }
    if (currpid != 0)
        close(socket_descriptor);
    return 0;
}