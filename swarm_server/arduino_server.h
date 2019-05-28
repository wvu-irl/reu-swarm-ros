#ifndef SERVER_HH
#define SERVER_HH

/**
 *
 * Server
 *
 * Author: Henry Vos
 *
 * Purpose:
 *   This section is to accept connections from all the robots.
 *   The clients will each get their own process. (this may pose issues)
 *	 The server will also pipe messages to the clients
 */


// Includes
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>

#include <functional>
#include <vector>

#define SERVER_PORT 4321 // port number

// datastructure that is used to send/recieve commands
// this is using a struct as it is only a public data storing application
typedef struct
{
  int rid;
  char str[64];
} command;

// sends a command indescriminantly to all the robots
void sendCommandToRobots(command cmd);

/**
 * Begins accepting connections to the server and processes commands from them
 *
 * command_callback is a registered function for taking care of command contents
 * info_callback is the registered function for printing basic information to the screen
 * error_callback is the registered function for showing server errors
 * exit_condition_callback is the registered function that dictates a successful exit condition
 * 													exits when exit_condition_callback() == false
 */
int beginServer(std::function<void(command)> command_callback,
                std::function<void(const char *, void *)> info_callback,
                std::function<void(const char *)> error_callback,
                std::function<bool()> exit_condition_callback);

#include "arduino_server_source.hpp"
#endif