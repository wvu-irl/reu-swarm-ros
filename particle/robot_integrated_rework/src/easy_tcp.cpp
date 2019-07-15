#include "easy_tcp.h"

EasyTCP::EasyTCP(void)
{
    client = TCPClient();
}

EasyTCP::EasyTCP(int _pt, byte _addr[], String _reg) : port(_pt), registerStr(_reg)
{
    client = TCPClient();
    address[0] = _addr[0];
    address[1] = _addr[1];
    address[2] = _addr[2];
    address[3] = _addr[3];
}

// Handles connection and registration. Returns false for a problem with either.
bool EasyTCP::init(int _timeout)
{
#if EASY_TCP_DEBUG
    Serial.println("TCP: Beginning init()");
#endif

    readTimer = 0;
    int currentTime = millis();
    int connect = -1;
    do
    {
        connect = client.connect(address, port);

#if EASY_TCP_DEBUG
        Serial.print("TCP: Trying to Connect : connect = ");
        Serial.println(connect);
#endif

    } while (connect != 1 && (millis() - currentTime) > CONN_TIMEOUT_MILLIS);

    //     int currentTime = millis();
    //    // waitFor(client.connected,_timeout); this doesn't work in a method, only in main...
    //     while (!client.connected() &&  millis()-currentTime> CONN_TIMEOUT_MILLIS)
    //     {
    // //         int x= (currentTime - millis());
    // // #if EASY_TCP_DEBUG
    // //         Serial.println("TCP: Trying to Connect, "+x);
    // // #endif
    //     }

    // Check if connection succeeded
    if (client.connected())
    {
#if EASY_TCP_DEBUG
        Serial.println("TCP: Connected.");
#endif
        Serial.println("register " + registerStr); // Append register identifier
        Serial.println();
        // Send registration to server (this dun work)
        // client.print("register ");
        // client.println(registerStr); // Append register identifier
        // client.println();

        client.println("register " + registerStr); // Append register identifier
        client.println();
        return true;
    }
    else
    {
#if EASY_TCP_DEBUG
        Serial.println("TCP: Connection timeout!");
#endif

        return false;
    }
}

// Just wraps the client's available()
int EasyTCP::available(void)
{
    return client.available();
}

//Wraps connected()
bool EasyTCP::connected(void)
{
    return client.connected();
}

// Identical to init, but can specify timeout. Not my best work.
bool EasyTCP::reconnect(int _timeout)
{

#if EASY_TCP_DEBUG
    Serial.println("TCP: Beginning reconnect()");
#endif

    readTimer = 0;

    client.connect(address, 4321);
    //waitFor(client.connected, _timeout);
    if (client.connected())
    {
#if EASY_TCP_DEBUG
        Serial.println("TCP: Connected.");
#endif

        // Send registration to server
        client.print("register ");
        client.println(registerStr); // Append register identifier
        client.println();

        return true;
    }
    else
    {
#if EASY_TCP_DEBUG
        Serial.println("TCP: Connection timeout!");
#endif

        return false;
    }
}

// Tries to read. If it fails, keeps track of time since last success.
//   Returns 0 if nothing available, -1 if error, # of bytes if success
// Also changes the vicon heading, passed by reference
int EasyTCP::read(uint8_t *_buf, size_t _len, float &_theta, float &_pos)
{

    if (readTimer == 0)
        readTimer = millis(); // Set timeout timer if first read

    // Loop while more than one packet is in buffer to clear it
    int bytes = client.available();
    while (bytes / _len > 1)
    {
        // Read to clear old packets from buffer
        client.read(_buf, _len);

        // Update bytes remainting in buffer
        bytes = client.available();
    }

#if EASY_TCP_DEBUG
    Serial.print("TCP: bytes = ");
    Serial.println(bytes);
#endif

    // Success case
    if (bytes > 0)
    {
        // Update last successful timestamp
        readTimer = millis();

        bytes = client.read(_buf, _len);
        _pos = (float)strtod(strtok((char *)_buf, ","), NULL); //i dunno how to get a str from _buf
        _theta = (float)strtod(strtok(NULL, ","), NULL);
    }
    // Error case
    else if (bytes < 0)
    {
#if EASY_TCP_DEBUG

        Serial.println("TCP: We somehow got negative bytes");
#endif
    }
    // No data case
    else
    {
        // If it's been too long with no data, try to reconnect
        if (millis() - readTimer >= READ_TIMEOUT_MILLIS)
        {
#if EASY_TCP_DEBUG
            Serial.println("TCP: Read timeout!");
#endif
            _theta = -1; // makes theta invalid
            // Disconnect
            client.stop();
            bool reConn = false;
            // Reconnect, timeout 1000ms
            while (!reConn)
            {
                init(CONN_TIMEOUT_MILLIS);
#if EASY_TCP_DEBUG
                if (!reConn)
                    Serial.println(reConn ? "TCP: Reconnected." : "TCP: Reconnect timeout!");
#endif
            }
        }
    }
    return bytes;
}