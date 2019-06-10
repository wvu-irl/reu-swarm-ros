#ifndef ARDINO_SERVER_SOURCE
#define ARDINO_SERVER_SOURCE
// definition of a "verbose" option
#define DEBUG_CPP 0

// setting this to 1 shows what messages failed and succeeded
#define DEBUG_MESSAGE 0

#include "arduino_server.h"

// struct not really useful to anything outside this file
// this struct is used to pass information between the main accept loop
// and the client processing thread
struct client_param
{
	std::function<void(command)> command_callback;
	std::function<void(const char *, void *)> info_callback;
	std::function<void(const char *)> error_callback;
	std::function<bool()> exit_condition_callback;
	int id;
};

// ConnectionInfo class implementation
ConnectionInfo::ConnectionInfo(int connection_descriptor)
{
#if DEBUG_CPP
	puts("SERVER (OBJ): Started connecting info");
	printf("SERVER (OBJ): descriptor exists: %d\n",
			&connection_descriptor != NULL);
#endif
	this->connection_descriptor = connection_descriptor;
	this->rid = -1;

#if DEBUG_CPP
	puts("SERVER (OBJ): Constructed");
#endif
}

// accessors
int ConnectionInfo::getRID()
{
	return this->rid;
}

void ConnectionInfo::setRID(int rid)
{
	this->rid = rid;
}

int ConnectionInfo::getConnectionDescriptor()
{
	return this->connection_descriptor;
}

// structure for all the socket descriptors
std::vector<ConnectionInfo> *sockets;

// map for quick access
std::map<int, ConnectionInfo> *registry;
std::vector<ConnectionInfo> *monitors;

void sendCommandToRobots(command cmd, int recip_rid)
{
#if DEBUG_CPP || DEBUG_MESSAGE
	//printf("[\033[1;33marduino_server_source\033[0m] Command executing\033[0m\n");
	if (registry->find(recip_rid)->first == recip_rid && registry->size() > 0)
		printf("\033[30;42mSERVER: sending message [%02d <-> %s]: %s\t%d\033[0m\n",
				recip_rid, rid_indexing[recip_rid].c_str(), cmd.str,
				registry->at(recip_rid).getConnectionDescriptor());
#endif

	// sending directly to recipiant
	if (registry->find(recip_rid)->first == recip_rid && registry->size() > 0) // check to see that the location in the map exists
		send(registry->at(recip_rid).getConnectionDescriptor(), &cmd, COMMAND_SIZE,
				0); // sending message
#if DEBUG_CPP || DEBUG_MESSAGE
	else
		printf(
				"\033[37;41mCould not locate ConnectionInfo for %02d <-> %s\033[0m\n",
				recip_rid, rid_indexing[recip_rid].c_str());
#endif

	// checking for monitors
	if (monitors->size() > 0)
	{
		char mon_str[64];
		// creating monitor message
		sprintf(mon_str, "[%02d / %s]:\t%s", recip_rid,
				rid_indexing[recip_rid].c_str(), cmd.str);
		strncpy(cmd.str, mon_str, sizeof(cmd.str)); // safe copy
#if DEBUG_CPP || DEBUG_MESSAGE
		printf("\033[37;44mSERVER: sending message to monitor: %s\033[0m\n",
				cmd.str);
#endif
		for (ConnectionInfo ci : *monitors) // sending message to all open monitors
		{
			send(ci.getConnectionDescriptor(), &cmd, COMMAND_SIZE, 0);
		}
	}
}

void sendCommandToRobots(command cmd)
{
	sendCommandToRobots(cmd, -1);
}

void *runClient(void *args)
{
#if DEBUG_CPP
	puts("Starting client thread");
#endif

	// getting parameters
	struct client_param *vals = (struct client_param *) args; // separating out parameter type

	// putting parameters into easily useabel variables
	std::function<void(command)> command_callback = vals->command_callback;
	std::function<void(const char *, void *)> info_callback = vals->info_callback;
	std::function<void(const char *)> error_callback = vals->error_callback;
	std::function < bool() > exit_condition_callback =
			vals->exit_condition_callback;
	int id = vals->id;

	// getting the connection_descriptor from the connection this thread is monitoring
	int connection_descriptor = sockets->at(id).getConnectionDescriptor();

	while (exit_condition_callback()) // running until told to stop
	{
		// Read data from connection into buffer. Continue to loop while message size is >0.
		int message_size = 0;
		command *buffer = ((command *) malloc(sizeof(command))); // allocating memory for read buffer
		//reading the message
		while ((message_size = read(connection_descriptor, buffer, sizeof(command)))
				> 0)
		{
			// Display if there was an error
			if (message_size == -1)
			{
				error_callback("Error receiving message.");
				exit(1);
			}

			// checking if the client is registering their RID
			if (strstr(buffer->str, "register") == buffer->str) // chekcing if the input starts with "register"
			{
				char num[2];

				sscanf(buffer->str, "register %s", num); // obtaining the ID
#if DEBUG_CPP
				printf("\033[34mAttempting to register: \033[37;%dm%s\033[0m\n",
						rid_map.find(std::string(num))->first == num ? 42 : 41, num);
#endif
				int rid =
						rid_map.find(std::string(num))->first == num ?
								rid_map.at(std::string(num)) : -1;

				sockets->at(id).setRID(rid); // setting the RID of the related object

				sockets->at(id).setRID(rid); // setting the RID of the related object
				if (rid == -2)
				{
					monitors->push_back(sockets->at(id));
				}
				else
				{
					registry->insert(
							std::pair<int, ConnectionInfo>(rid, sockets->at(id)));
#if DEBUG_CPP
					printf("SERVER: Registry size: \033[31m%d\033[0m\n",
							(int) registry->size());
#endif
				}

				info_callback("\033[34mRegistered %s\033[0m", (void *) (buffer->str));
			}
			// cheking to see if the exit command was sent
			else if (sockets->size() > 0
					&& strstr(buffer->str, "exit") == buffer->str)
			{
				ConnectionInfo leaving = sockets->at(id); // getting ConnectionInfo for the connection

				// removing if registered as a robot
				if (registry->find(leaving.getRID())->first == leaving.getRID()
						&& registry->size() > 0)
					registry->erase(leaving.getRID());

				// removing of registered as a monitor
				if (leaving.getRID() == -2)
				{
					int erase_id = 0;
					for (int i = 0; i < monitors->size(); i++)
					{
						if (monitors->size() > 0
								&& monitors->at(i).getConnectionDescriptor()
										== connection_descriptor) // matching connection descriptor
						{
							monitors->erase(monitors->begin() + i);
							break;
						}
					}
				}
			}
			// checking if the connection is doing a latency test
			else if (strstr(buffer->str, "ping") == buffer->str)
			{
				write(connection_descriptor, "pong", 4); // returns pong to sender
			}
			else
				command_callback(*buffer); // sending message to callback
		}

		free(buffer); // freeing buffer
	}
	pthread_exit(0); // exiting the client thread
}

int beginServer(std::function<void(command)> command_callback,
		std::function<void(const char *, void *)> info_callback,
		std::function<void(const char *)> error_callback,
		std::function<bool()> exit_condition_callback,
		std::function<void(const char *)> warn_callback)
{
#if DEBUG_CPP
	puts("SERVER: Getting socket");
#endif

	sockets = new std::vector<ConnectionInfo>();
	monitors = new std::vector<ConnectionInfo>();
	registry = new std::map<int, ConnectionInfo>();

	// starting socket
	int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_descriptor == -1)
	{
		error_callback("Error getting socket.");
		exit(1); // crash out
	}

	// Change receive timeout to 30 seconds.
	struct timeval timeout;
	timeout.tv_sec = 30;
	setsockopt(socket_descriptor, SOL_SOCKET, SO_RCVTIMEO,
			(struct timeval *) &timeout, sizeof(struct timeval));

	// Instantiate struct to store socket settings.
	struct sockaddr_in socket_address;
	memset(&socket_address, 0, sizeof(socket_address));
	socket_address.sin_family = AF_INET;
	socket_address.sin_addr.s_addr = INADDR_ANY;
	socket_address.sin_port = htons(SERVER_PORT);

#if DEBUG_CPP
	puts("SERVER: Binding socket");
#endif

	// Bind to the socket
	while (bind(socket_descriptor, (struct sockaddr *) &socket_address,
			sizeof(socket_address)) == -1)
	{
		char err[64];
		sprintf(err, "Error binding to socket (%d) retrying", errno); // making an error message that tells what went wrong
		// with binding the socket
		error_callback(err);
		throw "Binding failure";
		exit(1); // crash out
	}
#if DEBUG_CPP
	puts("SERVER: Listening to socket");
#endif
	// Set socket to listen for connections
	if (listen(socket_descriptor, 3) == -1)
	{
		error_callback("Error listening for connections.");
		exit(1); // crash out
	}

	int connection_descriptor; // variable that will contain the connection_descriptor of the most recent client accept
	struct sockaddr connection_addr;

	std::vector < pthread_t > threads; // vector to keep track of thread ids

#if DEBUG_CPP
	puts("SERVER: Starting socket loop");
#endif
	// Loop to handle connections on the socket
	while (exit_condition_callback())
	{
		// Specify struct to store information about accepted connection
		socklen_t connection_addr_size = sizeof(struct sockaddr);

#if DEBUG_CPP
		puts("SERVER: looking for accept");
#endif

		// Accept connection
		connection_descriptor = accept(socket_descriptor, &connection_addr,
				&connection_addr_size);
#if DEBUG_CPP
		puts("SERVER: accepted");
#endif
		sockets->push_back(ConnectionInfo(connection_descriptor));
#if DEBUG_CPP
		puts("SERVER: Made connection info object");
#endif
		if (connection_descriptor == -1)
		{
			warn_callback("Error accepting connection.");
			continue;
		}
		else
		{

			// collecting arguments for client thread
			struct client_param clinet_args = (struct client_param )
					{ command_callback, info_callback, error_callback,
							exit_condition_callback, (int) sockets->size() - 1 };

			pthread_attr_t attr;
			pthread_attr_init(&attr);
			pthread_t tid;

#if DEBUG_CPP
			puts("Starting thread");
#endif

			// starting client thread
			// this was done so that the server can keep accepting connections
			// as it is simultainiously communicating with a client
			pthread_create(&tid, &attr, runClient, &clinet_args);
			threads.push_back(tid); // keeping track of thread ids
		}
	}

	// waiting for all client handling to die
	for (pthread_t tid : threads)
	{
		pthread_join(tid, NULL); // waiting for threads to die
	}

	return 0;
}
#endif
