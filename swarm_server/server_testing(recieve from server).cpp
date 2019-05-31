
#include "arduino_server.h"

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

#include <sstream>
#include <string>

bool keep_alive = true;

void *sendThread(void *dat);

void info(const char *patt, void *dat)
{
  printf("SERVER INFO:");
  printf(patt, dat);
  printf("\n");
}

void handle(command patt)
{
  printf("PID: %d\tSERVER COMMAND: %s\n", getpid(), patt.str);
}

void eror(const char *str)
{
  printf("ERRROR: %s\n", str);
}

bool isOk()
{
  return keep_alive;
}

void runClient(int id)
{
  sleep(2);

  int socket_descriptor = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_descriptor == -1)
  {
    puts("Error getting socket.");
    exit(1);
  }

  // Instantiate struct to store socket settings.
  struct sockaddr_in socket_address;
  memset(&socket_address, 0, sizeof(socket_address));
  socket_address.sin_family = AF_INET;
  socket_address.sin_port = htons(4321);
  inet_aton("127.0.0.1", &(socket_address.sin_addr));

  // Connect to the socket
  if (connect(socket_descriptor, (struct sockaddr *)&socket_address, sizeof(socket_address)) == -1)
  {
    puts("Error connecting to socket.");
    exit(1);
  }

  command reg = {{'\0'}};
  sprintf(reg.str, "register %d", id);

  sleep(1);

  int bytes = write(socket_descriptor, (command *)(&reg), sizeof(command));
  if (bytes < 0)
  {
    puts("CLIENT ERROR: Failed registration");
    return;
  }

  while (true)
  {
    // TODO Recieve
    int message_size = 0;
    command buffer = {{'\0'}};

    int count = 0;

    //reading the message
    while ((message_size = read(socket_descriptor, &buffer, sizeof(buffer))) > 0)
    {
      printf("CLIENT %d GOT: %s\n", id, buffer.str);
    }

    // Display if there was an error
    if (message_size == -1)
    {
      puts("Error receiving message.");
      exit(1);
    }
  }
}

int main()
{
  pid_t curr_pid = getpid();
  int curr_ind = -1;
  for (size_t i = 0; i < 3 && curr_pid != 0; i++)
  {
    curr_ind++;
    curr_pid = fork();
    if (curr_pid < 0)
    {
      puts("Error forking processes");
      exit(1);
    }
  }
  puts("Clients created");
  if (curr_pid == 0)
    runClient(curr_ind);
  else
  {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_t tid;
    puts("Starting send thread");
    pthread_create(&tid, &attr, sendThread, NULL);
    puts("Starting server");
    beginServer(handle, info, eror, isOk);

    pthread_join(tid, NULL);
  }

  return 0;
}

void *sendThread(void *dat)
{
  sleep(5);

  for (size_t i = 0; i < sockets->size(); i++)
  {
    printf("CDT:\tCD:%d\tRID:%d\n", sockets->at(i).getConnectionDescriptor(), sockets->at(i).getRID());
  }

  puts("Sending commands");

  puts("Sending to 0");
  sendCommandToRobots((command){"Hello World"}, 0);

  puts("Sending command to 1");
  sendCommandToRobots((command){"dlroW olleH"}, 1);

  puts("Sending commands to all");
  sendCommandToRobots((command){"Everybody"}, -1);

  sleep(2);
  puts("Command sent");
  exit(0);
}