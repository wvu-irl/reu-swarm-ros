#ifndef EASY_TCP_H
#define EASY_TCP_H

#include <Particle.h>
#include <string.h>
#define EASY_TCP_DEBUG 0
#define CONN_TIMEOUT_MILLIS 30000
#define READ_TIMEOUT_MILLIS 5000

/*
 * EASY TCP
 * This class is meant to act as a wrapper for a TCPClient object.
 * It handles reading/writing, abstracting away the need to clear
 * outdated packets or worrying about buffers.
 */

typedef struct command {
    char str[32];
} command;

class EasyTCP {
public:
    EasyTCP(void);
    EasyTCP(int _pt, byte _addr[], String _reg);
    bool init(int _timeout = CONN_TIMEOUT_MILLIS);
    int available(void);
    int read(uint8_t* _buf, size_t _len);
private:
    TCPClient client;
    const int port = 4321;
    byte address[4] = {192, 168, 1, 187};
    const String registerStr = "XX";
    unsigned long readTimer;
};

#endif