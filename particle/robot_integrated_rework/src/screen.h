#ifndef SCREEN_H
#define SCREEN_H

// Color definitions
#define BLACK 0x0000
#define BLUE 0x001F
#define RED 0xF800
#define GREEN 0x07E0
#define CYAN 0x07FF
#define MAGENTA 0xF81F
#define YELLOW 0xFFE0
#define WHITE 0xFFFF

#include <string.h>
#include <Particle.h>
#include <Wire.h>
#include "spark_wiring.h"
#include <Adafruit_mfGFX.h>
#include <Adafruit_SSD1351_Photon.h>

enum BATT_STATE {B_NONE, B_LOW, B_MED, B_HIGH, B_CHARGING};

class Screen {
public:
    Screen(void);
    Screen(uint8_t cs, uint8_t rs, uint8_t sid, uint8_t sclk, uint8_t rst);

    void init(String _reg); // Turn on SSD1351

    
    void updateScreen(float _theta, bool _connected); //displays the information shown by the latter functions

    // TODO: need functions to output text, battery status, etc
    void sysStat(bool _connected); //shows the connection status
    void battStat(); //shows the battery status
    void oledArrow(float theta); //shows the direction the robot has been commanded to go

private:
    const uint8_t cs = A5, rs = D4, sid = D12, sclk = D13, rst = D5;
    Adafruit_SSD1351 oled = Adafruit_SSD1351(cs, rs, sid, sclk, rst);

    // Keeping track of status will allow us to only update pixels when needed, saving time
    enum BATT_STATE batteryState = B_NONE;
    bool tcpConnected = false;
    double oldTheta = 0.0;

    
    // TODO: variables for location of arrow / text cursor / etc?
};

#endif