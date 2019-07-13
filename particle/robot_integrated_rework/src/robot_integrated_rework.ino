#include <neopixel.h>
#include <MPU6050.h>
#include <Adafruit_mfGFX.h>
#include <Adafruit_SSD1351_Photon.h>

SYSTEM_THREAD(ENABLED)

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Wire.h>
#include "spark_wiring.h"
#include <Particle.h>

#include "screen.h"
#include "drivetrain.h"
#include "easy_tcp.h"
#include "imu_calibrate.h"
#define INTEGRATED_DEBUG 1

// // FOR AN ARGON BOARD
// #define mosi D12 //blue - DIN - MO on Argon Board
// #define sclk D13 //yellow
// #define cs A5    //orange
// #define dc D4    //green
// #define rst D5   //white

// IMPORTANT: Set pixel COUNT, PIN and TYPE
#define PIXEL_PIN D7
#define PIXEL_COUNT 1
#define PIXEL_TYPE WS2812B

#define NETWORK_LED 0

Adafruit_NeoPixel strip(PIXEL_COUNT, PIXEL_PIN, PIXEL_TYPE);

// Color definitions
#define BLACK 0x0000
#define BLUE 0x001F
#define RED 0xF800
#define GREEN 0x07E0
#define CYAN 0x07FF
#define MAGENTA 0xF81F
#define YELLOW 0xFFE0
#define WHITE 0xFFFF

Thread testThread; //("testThread", threadFunction);
void threadFunction(void);
DiffDrive diff_drive;
IMUCalibrate imu;
Screen screenObject;

float theta = 0, pos = 10;

struct command
{
    char str[32];
};

byte server[] = {192, 168, 10, 187};

void setup(void)
{
    Serial.begin(9600);
    waitUntil(WiFi.ready);
#if INTEGRATED_DEBUG
    Serial.println("Connected to wifi");
#endif
    // Initialize screen
    screenObject.init();

    // Set up pins
    // pinSetup();
    pinMode(PWR, INPUT);
    pinMode(CHG, INPUT);
    // disable on-board RGB LED on Photon/Electron / You can also set these pins for other uses. also kept this because idk what it is
    pinMode(RGBR, INPUT);
    pinMode(RGBG, INPUT);
    pinMode(RGBB, INPUT);

    // Initialize drivetrain
    diff_drive.init();

    // Initialize IMU
    imu.init();

    //neopixel stuff on hold until i get the reset working

    // Non-blocking replacement for delay(300), have no idea why it's here but i kept it
    int wait = millis() + 300;
    while (millis() < wait)
        ;

    // Initialize tcp client
    tcpClient.init(ip, port, "NE");
    oledThread = Thread("oled", threadFunction);
}

void loop()
{
    int temp = read(uint8_t _buf *, size_t _len, theta, float pos); // what are buf and len?
    if (temp > 0)
    {
       theta = imu.getIMUHeading();
#if INTEGRATED_DEBUG
        Serial.print("VICON\tR: ");
        Serial.print(pos);
        Serial.print("\tTH: ");
        Serial.print(theta);
#endif
    }
    else if (temp == 0)
    {
        imu.getIMUHeading();
#if INTEGRATED_DEBUG
        Serial.print("IMU\tR: ");
        Serial.print(pos);
        Serial.print("\tTH: ");
        Serial.print(theta);
#endif
    }
    else
    {
#if INTEGRATED_DEBUG
        Serial.println("Read Error!");
#endif
    }

    //need to get this from some decision betweeen imu and vicon
    diff_drive.drive(theta, pos);

    Serial.print("Finish drive command check client");
    Serial.println(millis());
    Particle.process();

    Particle.process();
}

void threadOled(void)
{
    while (true)
    {
        screenObject.updateScreen(theta, client.connected())
    }
}
#endif