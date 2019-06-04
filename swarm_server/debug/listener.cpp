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

#define PORT 4321
#define IP "127.0.0.1"

int main()
{
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
  socket_address.sin_port = htons(PORT);
  inet_aton(IP, &(socket_address.sin_addr));

  // Connect to the socket
  if (connect(socket_descriptor, (struct sockaddr *)&socket_address, sizeof(socket_address)) == -1)
  {
    puts("Error connecting to socket.");
    exit(1);
  }

  char reg[32] = "register YY";
  int bytes = write(socket_descriptor, (&reg), sizeof(reg));
  if (bytes < 0)
  {
    puts("Failed registration");
    return 1;
  }

  while (true)
  {
    int message_size = 0;
    char str[32];

    int count = 0;

    //reading the message
    while ((message_size = read(socket_descriptor, str, sizeof(str))) > 0)
    {
      printf("CLIENT GOT: %s\n", str);
    }

    // Display if there was an error
    if (message_size == -1)
    {
      puts("Error receiving message.");
      return 1;
    }
  }
  return 0;
}
