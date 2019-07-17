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
#include <swarm_server/robot_id.h>
#include <pthread.h>
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
#include <sys/wait.h>

#include <functional>
#include <vector>
#include <map>
#include <string>

#define SERVER_PORT 4321 // port number
#define COMMAND_SIZE 64

bool g_server_failure = false;

// datastructure that is used to send/recieve commands
// this is using a struct as it is only a public data storing application
typedef struct
{
	char str[COMMAND_SIZE];
} command;

/**
 *  Class contains all the data necessesary to talk with one robot
 *  as well as act as a filter with the rid to reduce bandwidth usage
 *
 */
class ConnectionInfo
{
private:
	int connection_descriptor; // the connection connection_descriptor for the socket
	int rid; // the id of the robot (or other client) connected

public:
	ConnectionInfo(int connection_descriptor);

	int getRID(); // returns the RID of the client
	void setRID(int rid); // sets the RID of the client

	int getConnectionDescriptor(); // returns the connection descriptor of the client
};

/**
 * Send command to robots sends commands to robots
 *
 * cmd is the content of the comannd
 *
 * recip_rid is the recipiant's RID
 *  special cases:
 *  a connection with registered id of -2 will recieve all sent commands
 *  a recip_rid of -1 will send a command to all robots
 */
void sendCommandToRobots(command cmd, int recip_rid);

// function responsible for recieving information from a client
void runClient(std::function<void(command, int)> command_callback,
		std::function<void(const char *, void *)> info_callback,
		std::function<void(const char *)> error_callback,
		std::function<bool()> exit_condition_callback, int id);

/**
 * Begins accepting connections to the server and processes commands from them
 *
 * command_callback is a registered function for taking care of command contents
 * info_callback is the registered function for printing basic information to the screen
 * error_callback is the registered function for showing server errors
 * exit_condition_callback is the registered function that dictates a successful exit condition
 * 													exits when exit_condition_callback() == false
 */
int beginServer(std::function<void(command, int)> command_callback,
		std::function<void(const char *, void *)> info_callback,
		std::function<void(const char *)> error_callback,
		std::function<bool()> exit_condition_callback,
		std::function<void(const char *)> warn_callback);
#include "arduino_server_source.cpp"
#endif
