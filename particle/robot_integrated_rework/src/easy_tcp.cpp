#include "easy_tcp.h"

EasyTCP::EasyTCP(void)
{
    //client = TCPClient();
}

EasyTCP::EasyTCP(int _pt, byte _addr[], String _reg) : port(_pt), registerStr(_reg)
{
    //client = TCPClient();
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

    client.connect(address, port);
    int currentTime = millis();
    while (!client.connected() && currentTime - millis() > CONN_TIMEOUT_MILLIS)
        ;

    // Check if connection succeeded
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
#if EASYTCP_DEBUG
    Serial.println("TCP: Beginning reconnect()");
#endif

    readTimer = 0;

    client.connect(server, 4321);
    waitFor(client.connected, _timeout);
    if (client.connected())
    {
#if EASYTCP_DEBUG
        Serial.println("TCP: Connected.");
#endif

        // Send registration to server
        client.print("register ");
        client.println(register); // Append register identifier
        client.println();

        return true;
    }
    else
    {
#if EASYTCP_DEBUG
        Serial.println("TCP: Connection timeout!");
#endif

        return false;
    }
}

// Tries to read. If it fails, keeps track of time since last success.
//   Returns 0 if nothing available, -1 if error, # of bytes if success
// Also changes the vicon heading, passed by reference
int EasyTCP::read(uint8_t _buf *, size_t _len, float &_theta, float &_pos)
{
#if TCP_DEBUG
    Serial.println("TCP: Beginning read()");
#endif

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

        bytes = client.read((uint8_t *)(&c), sizeof(struct command));
        _pos = (float)strtod(strtok(c.str, ","), NULL);
        _theta = (float)strtod(strtok(NULL, ","), NULL);
    }
    // Error case
    else if (bytes < 0)
    {
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

            // Disconnect
            client.stop();

            // Reconnect, timeout 1000ms
            bool reConn = init(1000);

#if EASY_TCP_DEBUG
            Serial.println(reConn ? "TCP: Reconnected." : "TCP: Reconnect timeout!");
#endif
        }
    }
    return bytes;
}