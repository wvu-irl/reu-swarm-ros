#include "easy_tcp.h"

EasyTCP::EasyTCP(void)
{
    client = TCPClient();
}

EasyTCP::EasyTCP(int _pt, byte _addr[4], string _reg) : port(_pt), address(_addr), register(_reg)
{
    client = TCPClient();
}

// Handles connection and registration. Returns false for a problem with either.
bool EasyTCP::init(void)
{
    #if TCP_DEBUG
    Serial.println("TCP: Beginning init()");
    #endif

    readTimer = 0;

    client.connect(server, 4321);
    waitFor(client.connected, CONN_TIMEOUT_MILLIS);
    if (client.connected())
    {
        #if TCP_DEBUG
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
        #if TCP_DEBUG
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

// Identical to init, but can specify timeout. Not my best work.
bool EasyTCP::reconnect(int _timeout)
{
    #if TCP_DEBUG
    Serial.println("TCP: Beginning reconnect()");
    #endif

    readTimer = 0;

    client.connect(server, 4321);
    waitFor(client.connected, _timeout);
    if (client.connected())
    {
        #if TCP_DEBUG
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
        #if TCP_DEBUG
        Serial.println("TCP: Connection timeout!");
        #endif

        return false;
    }
}

// Tries to read. If it fails, keeps track of time since last success.
//   Returns 0 if nothing available, -1 if error, # of bytes if success
int read(uint8_t _buf*, size_t _len)
{
    #if TCP_DEBUG
    Serial.println("TCP: Beginning read()");
    #endif

    if(readTimer == 0) readTimer = millis(); // Set timeout timer if first read

    // Loop while more than one packet is in buffer to clear it
    int bytes = client.available();
    while (bytes / _len > 1)
    {
        // Read to clear old packets from buffer
        client.read(_buf, _len);

        // Update bytes remainting in buffer
        bytes = client.available();
    }

    #if TCP_DEBUG
    Serial.print("TCP: bytes = ");
    Serial.println(bytes);
    #endif

    // Success case
    if(bytes > 0)
    {
        // Update last successful timestamp
        readMillis = millis();

        bytes = client.read((uint8_t *)(&c), sizeof(struct command));
    }
    // Error case
    else if(bytes < 0)
    {
    }
    // No data case
    else
    {
        // If it's been too long with no data, try to reconnect
        if(millis() - readMillis >= READ_TIMEOUT_MILLIS)
        {
            #if TCP_DEBUG
            Serial.println("TCP: Read timeout!");
            #endif

            // Disconnect
            client.disconnect();

            // Reconnect
            bool reConn = reconnect(1000);

            #if TCP_DEBUG
            Serial.println(reConn ? "TCP: Reconnected." : "TCP: Reconnect timeout!");
            #endif
        }
    }

    return bytes;
}