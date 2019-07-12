#include <neopixel.h>
#include <MPU6050.h>
#include <Adafruit_mfGFX.h>
#include <Adafruit_SSD1351_Photon.h>

SYSTEM_THREAD(ENABLED)
TCPClient client;

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Wire.h>
#include "spark_wiring.h"
#include <Particle.h>

// FOR AN ARGON BOARD
#define mosi D12 //blue - DIN - MO on Argon Board
#define sclk D13 //yellow
#define cs A5    //orange
#define dc D4    //green
#define rst D5   //white

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

float theta = 0, pos = 10;
float vicon_theta = 0, vicon_pos = 10;
float imu_theta = 0, imu_pos = 10;


// Pitch, Roll and Yaw values
float pitch = 0;
float roll = 0;
float yaw = 0.001;
float oldYaw = 0;
float timeStep = 0.001;
int t1=0;

struct command
{
    char str[32];
};

Servo myservoB, myservoA; //creating servo object

byte server[] = {192, 168, 10, 187};
// MPU variables:

/*
void handler(const char *topic, const char *data)
{
    Serial.println("received " + String(topic) + ": " + String(data));
}
*/

MPU6050 accelgyro;
int16_t ax, ay, az;
int16_t gx, gy, gz;

// Option 1: Hardware SPI - uses some analog pins, but much faster
//Adafruit_SSD1351 tft = Adafruit_SSD1351(cs, dc, rst);

// Option 2: Software SPI - use any pins but a little slower
Adafruit_SSD1351 tft = Adafruit_SSD1351(cs, dc, mosi, sclk, rst);

void setup(void)
{
    Serial.begin(9600);
    waitUntil(WiFi.ready);
    Serial.println("Connected to wifi");
    tft.begin();
    tft.fillScreen(BLACK);

    pinMode(PWR, INPUT);
    pinMode(CHG, INPUT);
    // disable on-board RGB LED on Photon/Electron / You can also set these pins for other uses.
    pinMode(RGBR, INPUT);
    pinMode(RGBG, INPUT);
    pinMode(RGBB, INPUT);
    RGB.onChange(ledChangeHandler);
    strip.begin();
    strip.show(); // Initialize all pixels to 'off'

    // Non-blocking replacement for delay(300)
    int wait = millis() + 300;
    while (millis() < wait)
        ;

    client.connect(server, 4321);
    waitFor(client.connected, 30000);
    if (client.connected())
    {
        Serial.println("Connected");
        client.println("register NE");
        client.println();
    }
    else
    {
        Serial.println("Connection failed");
    }

    myservoB.attach(A1); //attaching servos to pins
    myservoA.attach(A2);

    Wire.begin();

    Serial.println("Initializing I2C devices...");

    accelgyro.initialize();

    // Cerify the connection:
    Serial.println("Testing device connections...");
    Serial.println(accelgyro.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");

    testThread = Thread("test", threadFunction);
}

void loop()
{
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    Serial.println(gz);

    while (client.connected())
    {
        headingUpdate();
        Serial.print("\tR: ");
        Serial.print(pos);
        Serial.print("\tTH: ");
        Serial.print(theta);

        //Attempt at modifying pereira stuff to relative frame
        float v = 3 * cos(theta * 3.14 / 180);
        float w = 2 * sin(theta * 3.14 / 180);

        if (theta >= 0 && theta <= 90 || theta > 270)
        {
            myservoA.write(90 + 10 * (v - w));
            myservoB.write(90 - 10 * (v + w));
        }
        else if (theta > 90 && theta <= 180)
        { //we still want this because we want linear velocity to be non-negative ( change for pd)
            driveTurnCCW(theta);
        }
        else if (theta > 180 && theta < 270)
        {
            driveTurnCW(theta);
        }
        else//if somehow the command is invalid
        { 
            myservoA.write(90);
            myservoB.write(90); 
        }

        Serial.print("Finish drive command check client");
        Serial.println(millis());
        Particle.process();
    }

    if (!client.connected())
    {
        myservoA.write(90);
        myservoB.write(90);
        Serial.println("Disconnected.");
        client.connect(server, 4321);
        Serial.println("Reconnecting...");
        waitFor(client.connected, 30000);
    }
    Particle.process();
}

void threadFunction(void)
{
    tft.setTextSize(2);
    tft.setTextColor(WHITE);
    tft.println("Battery: ");
    tft.setTextSize(1);
    tft.setCursor(0, 28);
    tft.println("CONNECTIVITY: ");
    tft.drawTriangle(64, 38, 61, 43, 67, 43, WHITE); //Declares the front of the robot
    while (true)
    {
        oledArrow(theta);
    }
}

void driveTurnCW(float theta)
{
    myservoB.write(170);
    myservoA.write(110);
}

void driveTurnCCW(float theta)
{
    myservoB.write(70);
    myservoA.write(0);
}


void sysStat()
{
    if (client.connected())
    {
        tft.fillCircle(83, 31, 4, GREEN);
    }
    else
    {
        tft.fillCircle(83, 31, 4, RED);
    }
}

void battStat()
{
    tft.setCursor(0, 18);
    tft.setTextSize(1);
    float voltage = analogRead(BATT) * 0.0011224;
    if (voltage >= 4.2) // If battery voltage is greater than 4.2 V it is at full charge
    {
        tft.setTextColor(GREEN);
        tft.println("Full Charge");
    }
    else if (voltage < 4.2 or voltage > 3.5) // If battery voltage is between 4.2 and 3.5 V battery is at safe usage levels
    {
        tft.setTextColor(YELLOW);
        tft.println("Useable Level");
    }
    else if (voltage <= 3.5) // If battery voltage is less than 3.5 V battery needs charged
    {
        tft.setTextColor(RED);
        tft.println("Needs Charging");
    }
}


/*
void driveStat(float theta, float pos)
{
    tft.setTextSize(1);
    tft.setTextColor(WHITE);
    if (pos > 3)
    {
        tft.print("Driving to Target");
    }
    else if (pos < 3)
    {
        tft.print("Stop");
    }
}

void oledPrint(float theta, float pos)
{

    tft.setCursor(0, 5);
    tft.setTextSize(2);
    tft.println("Battery: ");
    Serial.println("BattStat start ");
    Serial.println(millis());

    battStat();
    Serial.println("battStat done, sysStat check ");
    Serial.println(millis());

    sysStat();
    Serial.println("sysStat done, updates ");
    Serial.println(millis());
    tft.setTextColor(WHITE);
    tft.setTextSize(2);
    tft.println("Theta:");
    tft.println(theta);

    tft.println("Radius:");
    tft.println(pos);
    driveStat(theta, pos);
    Serial.println("updates end ");
    Serial.println(millis());
    tft.fillRect(0, 20, 128, 24, BLACK);
    tft.fillRect(0, 60, 128, 15, BLACK);
    tft.fillRect(0, 91, 128, 128, BLACK);
    Serial.println("End ");
    Serial.println(millis());
}
*/

void oledArrow(float theta)
{
    tft.setCursor(0, 0);

    battStat();
    sysStat();

    tft.drawLine(64, 85, 64 + (40 * cos((theta + 90) * 3.14 / 180)), 85 - (40 * sin((theta + 90) * 3.14 / 180)), CYAN);
    tft.drawLine(64 + (40 * cos((theta + 90) * 3.14 / 180)), 85 - (40 * sin((theta + 90) * 3.14 / 180)), (64 + (30 * cos((theta + 100) * 3.14 / 180))), (85 - (30 * sin((theta + 100) * 3.14 / 180))), CYAN);
    tft.drawLine(64 + (40 * cos((theta + 90) * 3.14 / 180)), 85 - (40 * sin((theta + 90) * 3.14 / 180)), (64 + (30 * cos((theta + 80) * 3.14 / 180))), (85 - (30 * sin((theta + 80) * 3.14 / 180))), CYAN);
    tft.drawLine(64, 85, 64 + (40 * cos((theta + 90) * 3.14 / 180)), 85 - (40 * sin((theta + 90) * 3.14 / 180)), BLACK);
    tft.drawLine(64 + (40 * cos((theta + 90) * 3.14 / 180)), 85 - (40 * sin((theta + 90) * 3.14 / 180)), (64 + (30 * cos((theta + 100) * 3.14 / 180))), (85 - (30 * sin((theta + 100) * 3.14 / 180))), BLACK);
    tft.drawLine(64 + (40 * cos((theta + 90) * 3.14 / 180)), 85 - (40 * sin((theta + 90) * 3.14 / 180)), (64 + (30 * cos((theta + 80) * 3.14 / 180))), (85 - (30 * sin((theta + 80) * 3.14 / 180))), BLACK);
}

void ledChangeHandler(uint8_t r, uint8_t g, uint8_t b)
{
    strip.setPixelColor(NETWORK_LED, r, g, b, 0); // Change this if your using regular RGB leds vs RGBW Leds.
    strip.show();
}

/* 
void obstacleAvoid()
{
    myservoB.write(180);
    myservoA.write(0);
    delay(750);
    myservoB.write(0);
    myservoA.write(0);
    delay(500);
    myservoB.write(0);
    myservoA.write(180);
    delay(500);
}
*/

int getviconHeading()
{
    struct command c;

    // Loop while more than one packet is in buffer to clear it
    int bytes = client.available();
    while (bytes / sizeof(struct command) > 1)
    {
        // Read to clear old packets from buffer
        client.read((uint8_t *)(&c), sizeof(struct command));

        // Update bytes remainting in buffer
        bytes = client.available();
        Serial.print("x ");
    }
    bytes = client.read((uint8_t *)(&c), sizeof(struct command));

    // data parsing
    if (bytes > 0)
    {
        vicon_pos = (float)strtod(strtok(c.str, ","), NULL);
        vicon_theta = (float)strtod(strtok(NULL, ","), NULL);
    }

    return bytes;
}

void getIMUHeading()
{

    int t2 = millis();
    timeStep = t2 - t1;
    float oldYaw = gz / 131;
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    float yaw = gz / 131;

    imu_theta = theta - ((yaw + oldYaw) / 2) * timeStep * .001;
    t1 = t2;
}
void headingUpdate()
{
    getIMUHeading();
    if (getviconHeading() < 1)
    {
        Serial.println("imu heading");
        pos = imu_pos;
        theta = imu_theta;
    }
    else
    {
        Serial.println("Vicon Heading");
        pos = vicon_pos;
        theta = vicon_theta;
    }
}
