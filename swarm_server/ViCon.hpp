#pragma once
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

struct robot
{
    int id;
    double x;
    double y;
    double theta;
};


// sends an SQL query to the database
struct robot query(char *str, int sock_desc)
{
    int numSent = send(sock_desc, str, sizeof(char) * strlen(str), NULL);
}

// processes commands that are realated to the ViCon system
void processCommand(char *str, int sock_desc)
{
    // update command
    if (strstr("update", str) == str)
    {
        // recieving robot ids that have moved
        char data[256];
        sscanf(str, "update [%s]", data);

        // variable for id collection
        int ids[50] = { -1 };

        // separating out all the ids that are in the string
        char *val = strtok(str, ",");
        sscanf(val, "%d", ids[0]);

        for (size_t i = 1; (val = strtok(NULL, ",")) != NULL && i < 50; i++)
        {
            sscanf(val, "%d", ids[i]);
        }
        
        // querying all the data from the SQL database
        struct robot robData[50] = { -1 };
        // SELECT * FROM <TBD> WHERE rid = `id`
        for (int i = 0;i < 50 && ids[i] != -1;i++)
        {
            if (ids[i] != -1)
            {
                char str[64];
                sprintf(str, "SELECT * FROM db WHERE rid = %d;", ids[i]);
                robData[i] = query(str, sock_desc);
            }
        }
    }
}