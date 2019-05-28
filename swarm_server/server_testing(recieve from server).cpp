
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
#include <thread>

void *sendThread(void *dat);

void info(const char *patt, void *dat)
{
  std::ostringstream os;
  os << "SERVER INFO: " << patt << "\n";
  std::string full_pattern = os.str();
  const char *ch_pathh = full_pattern.c_str();
  printf(ch_pathh, dat);
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
  return true;
}

void runClient()
{
  sleep(1);

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

  while (true)
  {
    // TODO Recieve
    int message_size = 0;
    command buffer = { 0, {'\0'}};

    int count = 0;

    //reading the message
    while ((message_size = read(socket_descriptor, &buffer, sizeof(buffer))) > 0)
    {
      puts("Message recievd");
      printf("CLIENT GOT: %s\n", buffer.str);
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

  for (size_t i = 0; i < 50; i++)
  {
    if (curr_pid != 0)
    {
      curr_pid = fork();
    }
    else if (curr_pid < 0)
    {
      puts("Error forking processes");
      exit(1);
    }
  }

  if (curr_pid == 0)
    runClient();
  else
  {
    pthread_attr_t attr;
  	pthread_attr_init(&attr);
  	pthread_t tid;
  	pthread_create(&tid, &attr, sendThread, NULL);

    beginServer(handle, info, eror, isOk);
    
    pthread_join(tid, NULL);
  }

  return 0;
}

void *sendThread(void *dat)
{
  sleep(2);
  puts("Sending command");
  sendCommandToRobots((command){0, "Hello World"});
  puts("Command sent");
}