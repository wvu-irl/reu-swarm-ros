#ifndef ARDINO_SERVER_SOURCE
#define ARDINO_SERVER_SOURCE
#define DEBUG_CPP 0

#include "arduino_server.h"

// structure for all the socket descriptors
std::vector<int> *sockets;

void sendCommandToRobots(command cmd)
{
#if DEBUG_CPP
  printf("Sending message %s\n", cmd.str);
#endif
  for (size_t i = 0; i < sockets->size(); i++)
  {
#if DEBUG_CPP
    printf("Sending to %d\n", sockets->at(i));
#endif
    send(sockets->at(i), &cmd, sizeof(cmd), 0);
  }
}


int beginServer(std::function<void(command)> command_callback,
                std::function<void(const char *, void *)> info_callback,
                std::function<void(const char *)> error_callback,
                std::function<bool()> exit_condition_callback)
{
  // creating shared memory for all the different socket descriptors
  // Create variables for file descriptor.
	int fd = -1;
	// Open /dev/zero as read-write to use so values are initialized.
	if ((fd = open("/dev/zero", O_RDWR)) == -1)
	{
		error_callback("Could not open /dev/zero for shared memory.");
		exit(1);
	}
  // allocating shared memory
	sockets = (std::vector<int> *)mmap(NULL, sizeof(std::vector<int>), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);


  // starting socket
  int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_descriptor == -1)
  {
    error_callback("Error getting socket.");
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
  socket_address.sin_port = htons(SERVER_PORT);

  // Bind to the socket
  if (bind(socket_descriptor, (struct sockaddr *)&socket_address, sizeof(socket_address)) == -1)
  {
    error_callback("Error binding to socket.");
    exit(1);
  }

  // Set socket to listen for connections
  if (listen(socket_descriptor, 3) == -1)
  {
    error_callback("Error listening for connections.");
    exit(1);
  }

  pid_t currpid = getpid();
  int connection_descriptor;
  struct sockaddr connection_addr;
  int num_conns = 0;
  // Loop to handle connections on the socket
  while (exit_condition_callback())
  {
    if (currpid != 0)
    {
      // Specify struct to store information about accepted connection
      socklen_t connection_addr_size = sizeof(struct sockaddr);

      // Accept connection
      connection_descriptor = accept(socket_descriptor, &connection_addr, &connection_addr_size);
      sockets->push_back(connection_descriptor);
      if (connection_descriptor == -1)
      {
        error_callback("Error accepting connection.");
        continue;
      }
      else
      {
        num_conns++;
        currpid = fork(); // creating a client handling process
        if (currpid == 0)
        { // one time run / setup for client handeler
          char *client_ip = inet_ntoa(((struct sockaddr_in *)&connection_addr)->sin_addr);
          info_callback("Connected to %s:4321", client_ip);
        }
      }
    }
    else // in client handler
    {
      // handeling messages from the client
      // Get char representation of client IP address
      char *client_ip = inet_ntoa(((struct sockaddr_in *)&connection_addr)->sin_addr);

      // Read data from connection into buffer. Continue to loop while message size is >0.
      int message_size = 0;
      command *buffer = ((command *)malloc(sizeof(command)));

      int count = 0;

      //reading the message
      while ((message_size = read(connection_descriptor, &buffer, sizeof(buffer))) > 0)
      {
        command_callback(*buffer); // sending message to callback
      }

      // Display if there was an error
      if (message_size == -1)
      {
        error_callback("Error receiving message.");
        exit(1);
      }

      free(buffer);
    }
  }
  // only closes the socket if the process is the parent
  if (currpid != 0)
  {
    close(socket_descriptor); // closing socket
  }
  return 0;
}
#endif