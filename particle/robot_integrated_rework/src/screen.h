#ifndef SCREEN_H
#define SCREEN_H

#include <Particle.h>
#include <Wire.h>
#include "spark_wiring.h"
#include <Adafruit_mfGFX.h>
#include <Adafruit_SSD1351_Photon.h>

class Screen {
public:
    Screen(void);
    Screen(uint8_t cs, uint8_t rs, uint8_t sid, uint8_t sclk, uint8_t rst);

    void init(void); // Turn on SSD1351
    // TODO: need functions to output text, battery status, etc
private:
    Adafruit_SSD1351 oled;
    const uint8_t cs = A5, rs = D4, sid = D12, sclk = D13, rst = D5;
    // TODO: variables for location of arrow / text cursor / etc?
};

#endif