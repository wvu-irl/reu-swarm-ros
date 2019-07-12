#ifndef EASY_TCP_H
#define EASY_TCP_H

#include <Particle.h>

class EasyTCP {
public:
    EasyTCP(void);
private:
    TCPClient client;
    const int port;
    const byte address[4];
};

#endif