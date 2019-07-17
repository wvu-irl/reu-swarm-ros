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
#include "charger_status.h"
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
String registerString = "MA";
EasyTCP tcpClient(port, ip, registerString);
  struct command c;
float theta = -100, pos = 10;

ChargerStatus chargeStat;
unsigned long lastBattPublish;
String prevMessage;
String message;

void setup(void)
{
    Serial.begin(9600);
    waitUntil(WiFi.ready);

#if INTEGRATED_DEBUG
    Serial.println("Connected to wifi");
#endif

    // Initialize screen
    screenObject.init(registerString);

    //Send it's first battery status then intialize the lastPublish for when it last published status
    prevMessage=chargeStat.checkChargingState();
    tcpClient.println(prevMessage);
    lastBattPublish=millis();

    

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

    // TODO: neopixel stuff on hold until i get the reset working

    // Initialize tcp client
    while(!tcpClient.init(10000)) tone(A4, 1760, 1000); // screams out an A6 on pin A4 :^)
    oledThread = Thread("oled", threadOled);
}

void loop()
{
    if (!tcpClient.connected()) {
        // Stop moving
        diff_drive.fullStop();

        // Keep trying to reconnect as needed
        while(!tcpClient.init(10000)) tone(A4, 1760, 1000);

        // Restart drivetrain
        diff_drive.restart();
    }

    // Read from TCP
    int temp = tcpClient.read((uint8_t *)(&c), sizeof(struct command), theta, pos); 

    // If got a heading from VICON, just update IMU calibration
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
    // If no data, use IMU estimate
    else if (temp == 0)
    {
        theta = imu.getIMUHeading(theta);
#if COMMAND_DEBUG
        Serial.print("IMU\tR: ");
        Serial.print(pos);
        Serial.print("\tTH: ");
        Serial.print(theta);
#endif
    }
    // Else, error
    else
    {
#if INTEGRATED_DEBUG
        Serial.println("Read Error!");
#endif
    }
    //need to get this from some decision betweeen imu and vicon
   
    diff_drive.drive(theta, pos, imu.getYawRate());
    Serial.print("Finish drive command check client ");
    Serial.println(millis());
    
    message=chargeStat.checkChargingState();
    if(!message.equals(prevMessage)) // Send it
        tcpClient.println(message);
        prevMessage=message;
    if(millis()-lastBattPublish>60000)
        tcpClient.println(chargeStat.giveBatteryVoltage());


    

    Particle.process();
    
}

void threadOled(void)
{
    while (true)
    {
        screenObject.updateScreen(theta, tcpClient.connected(),tcpClient.readTimer);
    }
}