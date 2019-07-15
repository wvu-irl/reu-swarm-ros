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
#define COMMAND_DEBUG 0
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

typedef struct command {
    char str[32];
} command;

Thread oledThread; //("testThread", threadFunction);
void threadOled(void);
DiffDrive diff_drive;
IMUCalibrate imu;
Screen screenObject;
 int port = 4321;
     byte ip[4] = {192, 168, 10, 187};
EasyTCP tcpClient( port,ip, "NH");
  struct command c;
float theta = 0, pos = 10;

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
    Wire.begin();
    // Initialize drivetrain
    diff_drive.init();

    // Initialize IMU
    imu.init();

    //neopixel stuff on hold until i get the reset working


    // Non-blocking replacement for delay(3000), have no idea why it's here but i kept it
    // int wait = millis() + 3000;
    // while (millis() < wait){
    //     Serial.println("waiting");
    // }


    // Initialize tcp client
    while(!tcpClient.init(10000));
    oledThread = Thread("oled", threadOled);
}

void loop()
{
    int temp = tcpClient.read((uint8_t *)(&c), sizeof(struct command), theta, pos); // what are buf and len?
    if (temp > 0)
    {
        imu.getIMUHeading(theta);
#if COMMAND_DEBUG
        Serial.print("VICON\tR: ");
        Serial.print(pos);
        Serial.print("\tTH: ");
        Serial.print(theta);
#endif
    }
    else if (temp == 0)
    {
        theta= imu.getIMUHeading(theta);
#if COMMAND_DEBUG
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
    if (temp<0) while(!tcpClient.init(10000));
    Serial.print("Finish drive command check client");
    Serial.println(millis());
    Particle.process();
}

void threadOled(void)
{
    while (true)
    {
        screenObject.updateScreen(theta, tcpClient.connected());
    }
}