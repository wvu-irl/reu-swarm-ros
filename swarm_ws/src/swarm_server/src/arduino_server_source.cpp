#ifndef ARDINO_SERVER_SOURCE
#define ARDINO_SERVER_SOURCE
// definition of a "verbose" option
#define DEBUG_CPP 0

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
  printf("SERVER (OBJ): descriptor exists: // crash out %d\n", &connection_descriptor != NULL);
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

void sendCommandToRobots(command cmd, int recip_rid)
{
	for (size_t i = 0; i < sockets->size(); i++) // checking every logged connection
	{
		if (sockets->at(i).getRID() == -2 || sockets->at(i).getRID() == recip_rid
				|| recip_rid == -1) // selecting who is to be send the command
			send(sockets->at(i).getConnectionDescriptor(), &cmd, sizeof(cmd), 0); // sending the command
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
				int rid = (int) strtol(num, NULL, 10);

				sockets->at(id).setRID(rid); // setting the RID of the related object

				info_callback("Registered %s", (void *) (rid_indexing[rid].c_str()));
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
		std::function<bool()> exit_condition_callback)
{
#if DEBUG_CPP
    puts("SERVER: Getting socket");
#endif

	sockets = new std::vector<ConnectionInfo>();

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
	if (bind(socket_descriptor, (struct sockaddr *) &socket_address,
			sizeof(socket_address)) == -1)
	{
		char err[32];
		sprintf(err, "Error binding to socket %d", errno); // making an error message that tells what went wrong
																											 // with binding the socket
		error_callback(err);
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
			error_callback("Error accepting connection.");
			continue;
		}
		else
		{
#if DEBUG_CPP
        puts("SERVER: getting client IP");
#endif
			// getting client IP for printing
			char *client_ip = inet_ntoa(
					((struct sockaddr_in *) &connection_addr)->sin_addr);
			info_callback("Connected to %s:4321", client_ip);

			// collecting arguments for client thread
			struct client_param clinet_args = (struct client_param
					)
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
