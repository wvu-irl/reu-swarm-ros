#ifndef EASY_TCP_H
#define EASY_TCP_H

#include <Particle.h>
#define EASY_TCP_DEBUG 1
#include <string.h>
#define CONN_TIMEOUT_MILLIS 10000
#define READ_TIMEOUT_MILLIS 1000

/*
 * EASY TCP
 * This class is meant to act as a wrapper for a TCPClient object.
 * It handles reading/writing, abstracting away the need to clear
 * outdated packets or worrying about buffers.
 */



class EasyTCP {
public:
    EasyTCP(void);
    EasyTCP(int _pt, byte _addr[], String _reg);
    bool init(int _timeout = CONN_TIMEOUT_MILLIS);
    int available(void);
    bool connected(void);
      unsigned long readTimer;
    int read(uint8_t *_buf, size_t _len, float &_theta, float &_pos, char *sys_comm);
    void write(uint8_t *_buf, size_t _len); //UNIMPLEMENTED
    void println(String _s);
    void disconnect(void);
private:
    TCPClient client;
    const int port = 4321;
     byte address[4] = {192, 168, 10, 187};
    String registerStr = "PA";
  
};

#endif