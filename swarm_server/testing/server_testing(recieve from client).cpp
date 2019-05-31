
#include "../arduino_server.h"

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
	printf("PID: %d\tSERVER COMMAND: %s\n",getpid(), patt.str);
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

	sleep(1);

	command cmds[3];
	strcpy(cmds[0].str, "test0");
	strcpy(cmds[1].str, "test1");
	strcpy(cmds[2].str, "test2");

	for (size_t i = 0; i < sizeof(cmds) / sizeof(command); i++)
	{
		int bytes = write(socket_descriptor, cmds + i, sizeof(cmds[i]));
		if (bytes == -1)
		{
			puts("CLIENT ERROR: write from client failed");
		}
	}

	close(socket_descriptor);
	return;
}

int main()
{
	pid_t curr_pid = getpid();

	for (size_t i = 0; i < 2; i++)
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
		beginServer(handle, info, puts, isOk);

	return 0;
}
