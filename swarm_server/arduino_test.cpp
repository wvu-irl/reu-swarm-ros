#include "arduino_server.h"
#include <sstream>
#include <string>

void commandCallback(command cmd)
{
    printf("Robot[%d] head %s\n", 0, "asdg");
}

void info(const char *patt, void *dat)
{
    std::ostringstream os;
    os << "SERVER INFO: " << patt << "\n";
    std::string full_pattern = os.str();
    const char *ch_pathh = full_pattern.c_str();
    printf(ch_pathh, dat);
}

bool keepalive()
{
    return true;
}

void *sendThread(void *arg0)
{
    sleep(30);
    puts("Sending messages");

    while (true)
    {
        command output = {"0,0.5,135.4"};
        sendCommandToRobots(output);
        sleep(1);
    }
}

int main()
{
    puts("Starting");
    printf("Sending %d bytes per message\n", (int)sizeof(command));

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_t tid;
    pthread_create(&tid, &attr, sendThread, NULL);

    beginServer(commandCallback, info, puts, keepalive);

    pthread_join(tid, NULL);
}