/******************************************************/
//       THIS IS A GENERATED FILE - DO NOT EDIT       //
/******************************************************/

#line 1 "/home/smart2/git/reu-swarm-ros/robot_integrated_rework/src/robot_integrated_rework.ino"
// This #include statement was automatically added by the Particle IDE.
//#include <pid.h>

// This #include statement was automatically added by the Particle IDE.
#include <I2Cdev.h>

// This #include statement was automatically added by the Particle IDE.




// This #include statement was automatically added by the Particle IDE.
#include <neopixel.h>
// This #include statement was automatically added by the Particle IDE.
#include <MPU6050.h>
// This #include statement was automatically added by the Particle IDE.
#include <Adafruit_mfGFX.h>
// This #include statement was automatically added by the Particle IDE.
//#include "MPU6050_6Axis_MotionApps20.h"
#include <Adafruit_SSD1351_Photon.h>
#include "Adafruit_mfGFX/Adafruit_mfGFX.h"

SYSTEM_THREAD(ENABLED)
TCPClient client;

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Wire.h>
#include "spark_wiring.h"
#include <Particle.h>
void handler(const char *topic, const char *data);
void setup(void);
void loop();
void driveTurnCW(float theta);
void driveTurnCCW(float theta);
void driveStraight(float vel);
void sysStat();
void battStat();
void driveStat(float theta, float pos);
void oledPrint(float theta, float pos);
void oledArrow(float theta);
void ledChangeHandler(uint8_t r, uint8_t g, uint8_t b);
void obstacleAvoid();
int getviconHeading();
void getIMUHeading();
void headingUpdate();
#line 32 "/home/smart2/git/reu-swarm-ros/robot_integrated_rework/src/robot_integrated_rework.ino"


// FOR AN ARGON BOARD
#define mosi D12 //blue - DIN - MO on Argon Board
#define sclk D13 //yellow
#define cs   A5 //orange
#define dc   D4 //green
#define rst  D5 //white

// IMPORTANT: Set pixel COUNT, PIN and TYPE
#define PIXEL_PIN D7
#define PIXEL_COUNT 1
#define PIXEL_TYPE WS2812B

#define NETWORK_LED 0

Adafruit_NeoPixel strip(PIXEL_COUNT, PIXEL_PIN, PIXEL_TYPE);


// You can use any 5 pins; see note below about hardware vs software SPI - FOR PHOTON!!!!!!!!!!!
/*/FOR A PHOTON BOARD
#define cs   D5
#define sclk A3
#define mosi A5
#define rst  D4
#define dc   D6
*/
// Color definitions
#define	BLACK           0x0000
#define	BLUE            0x001F
#define	RED             0xF800
#define	GREEN           0x07E0
#define CYAN            0x07FF
#define MAGENTA         0xF81F
#define YELLOW          0xFFE0
#define WHITE           0xFFFF

MPU6050 mpu;
// PID definitions
//#define PIN_INPUT 0
//#define PIN_OUTPUT 3

// PID variables
double Setpoint, Input, Output;

// PID initial tuning parameters
double Kp=2, Ki=5, Kd=1;
//PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, PID::DIRECT);
/* Input: varible we are trying to control (heading/yaw read from IMU)
Output: variable that will be adjusted by PID (Power to servos)
Setpoint: value we want to maintain (desired heading from VICON)  */
//PID myPID(&Input, &Output, &Setpoint,1,0,1, PID::REVERSE);

bool turn;

Thread testThread;//("testThread", threadFunction);
void threadFunction(void);


float theta = 0, pos = 10;
float vicon_theta = 0, vicon_pos = 10;
float imu_theta = 0, imu_pos = 10;

unsigned long timer = 0;
float timeStep = 0.001;
int t1=0;

// Pitch, Roll and Yaw values
float pitch = 0;
float roll = 0;
float yaw = 0.001;
float oldYaw = 0;




// //mpu DMP
// bool dmpReady = false;  // set true if DMP init was successful
// uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
// uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
// uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
// uint16_t fifoCount;     // count of all bytes currently in FIFO
// uint8_t fifoBuffer[64]; // FIFO storage buffer

// // orientation/motion vars
// Quaternion q;           // [w, x, y, z]         quaternion container
// VectorInt16 aa;         // [x, y, z]            accel sensor measurements
// VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
// VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
// VectorFloat gravity;    // [x, y, z]            gravity vector
// float euler[3];         // [psi, theta, phi]    Euler angle container


// float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector


 float error=0;
 float oldError=0;
 float dError=0;
 float omega=0;
 float kp=1;
 float kd=.07;
 float aCommand=90;
 float bCommand=90;
 float tStep=.01;




struct command {
  char str[32];
};

struct heading {
  float r;
  float theta;
};

Servo myservoB, myservoA; //creating servo object

byte server[] = { 192, 168, 10, 187 };
// MPU variables:


void handler(const char *topic, const char *data) {
    Serial.println("received " + String(topic) + ": " + String(data));
}

MPU6050 accelgyro;
int16_t ax, ay, az;
int16_t gx, gy, gz;

// Option 1: Hardware SPI - uses some analog pins, but much faster
//Adafruit_SSD1351 tft = Adafruit_SSD1351(cs, dc, rst);

// Option 2: Software SPI - use any pins but a little slower
Adafruit_SSD1351 tft = Adafruit_SSD1351(cs, dc, mosi, sclk, rst);



void setup(void)
{
  //WiFi.on() ;
  //WiFi.connect();
  //for(int i  = 0; i < 30000 && WiFi.connecting(); i++) delay(1);
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
   while(millis() < wait);


  // Make sure your Serial Terminal app is closed before powering your device
   //Particle.subscribe("particle/device/ip", handler);
   //Particle.publish("particle/device/ip");
  // Wait for a USB serial connection for up to 30 seconds
 // waitFor(Serial.isConnected, 30000);

    // PID initialization
    //Input = analogRead(PIN_INPUT); // Sensor input pin ()
    //Setpoint = 0; // Desired setpoint (heading goal from vicon?)


    // Activate PID
    //\myPID.SetMode(PID::AUTOMATIC);
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

    // The following line will wait until you connect to the Spark.io using serial and hit enter. This gives
    // you enough time to start capturing the data when you are ready instead of just spewing data to the UART.
    //
    // So, open a serial connection using something like:
    // screen /dev/tty.usbmodem1411 9600
    //while(!Serial.available()) SPARK_WLAN_Loop();

    Serial.println("Initializing I2C devices...");
    // //waitUntil(client.connected);


   accelgyro.initialize();

    // Cerify the connection:
   Serial.println("Testing device connections...");
    Serial.println(accelgyro.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");


    testThread = Thread("test", threadFunction);
}


void loop() {
//   if(client.status()){
//       Serial.println("Client is connected");
//   }
//   if (client.available()){
//       Serial.println("Client is Available");
//   }
//   else if(!client.available()){
//       Serial.println("Client Not Available");
//   }
 accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        Serial.println(gz);

  while (client.connected())
  {
        //Serial.print("Ping server update");
        //Serial.println(millis());

        headingUpdate();
        Serial.print("\tR: ");
        Serial.print(pos);
        Serial.print("\tTH: ");
        Serial.print(theta);

        //Serial.println(sizeof(struct command));
       /* Serial.print("oled update");
        Serial.println(millis());
        oledArrow(theta);
        Serial.print("finished oled, request imu");
        Serial.println(millis());
        */
        //accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        //Serial.println(gz);

        /*
        if (theta> 20 && theta<180){
          driveTurnCCW(theta);
          Serial.print("ccw");
        }
        else if(theta>=180 && theta<340){
          driveTurnCW(theta);
          Serial.print("cw");
        }
        else if (theta< 20 && theta>=340 &&  pos >3){
          driveStraight(pos);
          Serial.print("straight");
        }
        else if (theta <20 && theta>=340 && pos < 3){
          myservo.write(90);
          myservo2.write(90);
          Serial.print("stop");
        }
        */

   // Serial.print("Start drive command");
    //Serial.println(millis());
    //imu collision
    // if(abs(ax)>20000 or abs(ay)>20000){
    //     obstacleAvoid();
    // }

    //jank control but works
    // if (theta>=0 && theta<=90){
    //  myservoA.write(90+10*(4-5*theta/90));
    //   myservoB.write(90-10*(4-3*theta/90));
    // }
    // else if(theta>=270 && theta<360){
    //  myservoA.write(90+10*(4-3*(360-theta)/90));
    //  myservoB.write(90-10*(4-5*(360-theta)/90));
    // }
    // else if(theta>90 && theta<=180){
    //     driveTurnCCW(theta);
    // }
    // else if(theta>180 && theta<270){
    //     driveTurnCW(theta);
    // }

    //Attempt at modifying pereira stuff to relative frame
    //float v = 4*cos(theta*3.14/180);
    //float w = 1*sin(theta*3.14/180);

    if (theta>=0 && theta<=90 || theta>270){
        float tstep2=millis();

        float l=((bCommand)-(aCommand)/yaw);
        error = atan2(sin(theta),cos(theta));
        dError=(error-oldError)/tStep;
        omega=kp*error+kd*dError;
        if(theta<180){
        aCommand=-(omega/l)+bCommand;
        }
        else{
        bCommand=(omega/l)+aCommand;
        }
    //Acommand=90+10*(v-w);
    //Bcommand=90-10*(v+w);

        myservoA.write(aCommand+90); //180
        myservoB.write(bCommand-90); //0
        float tstep1=tstep2;
        tStep=tstep2-tstep1;
    }
    //else if(theta>90 && theta<=180){ //we still want this because we want linear velocity to be non-negative ( change for pd)
      //  driveTurnCCW(theta);

    else if(theta>90 && theta<=180){ //we still want this because we want linear velocity to be non-negative ( change for pd)
        driveTurnCCW(theta);
    }
    else if(theta>180 && theta<270){
    driveTurnCW(theta);
    }


    /* PID
    Input = analogRead(PIN_INPUT);  // input = imu read for current yaw
    myPID.Compute();                // perform PID algorithm
    analogWrite(PIN_OUTPUT, Output);// output = servo command to correct yaw
    */

    Serial.print("Finish drive command check client");
    Serial.println(millis());
    Particle.process();


  }

  if (!client.connected()) {
    myservoA.write(90);
    myservoB.write(90);
    Serial.println("Disconnected.");
    client.connect(server, 4321);
    Serial.println("Reconnecting...");
    waitFor(client.connected, 30000);

  }
  Particle.process();
}


void threadFunction(void) {
    tft.setTextSize(2);
    tft.setTextColor(WHITE);
    tft.println("Battery: ");
    tft.setTextSize(1);
    tft.setCursor(0, 28);
    tft.println("CONNECTIVITY: ");
    tft.drawTriangle(64,38,61,43,67,43,WHITE); //Declares the front of the robot
	while(true) {
		oledArrow(theta);
	}
}

void driveTurnCW(float theta){
    myservoB.write(170);
    myservoA.write(110);
    delay(10);
}


void driveTurnCCW(float theta){
    myservoB.write(70);
    myservoA.write(0);
    delay(10);
}

void driveStraight(float vel){
    myservoA.write(180);
    myservoB.write(0);
    delay(20);
}
// void driveTurn(){
//     if (theta>180){
//       Input = map((int)theta, 180, 360, -180, 0);
//         }
//      else {
//       Input=(int)theta;
//   }

//     myPID.Compute();
//     myservoB.write(90+90*Output);
//     myservoA.write(90-90*Output);
//     delay(1);
//     if(theta<10||theta>350){
//         turn==false;
//     }
// }

void sysStat(){

    if (client.connected())
    {
        tft.fillCircle(83,31,4,GREEN);

    }
    else
    {
        tft.fillCircle(83,31,4,RED);
    }
}


void battStat() {
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
    else if (voltage <= 3.5)                        // If battery voltage is less than 3.5 V battery needs charged
    {
        tft.setTextColor(RED);
        tft.println("Needs Charging");
    }

}

void driveStat(float theta, float pos){
    tft.setTextSize(1);
    tft.setTextColor(WHITE);
    if (pos>3){
        tft.print("Driving to Target");
    }
    else if(pos<3){
        tft.print("Stop");
    }
}

void oledPrint(float theta, float pos) {

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
    tft.fillRect(0,20,128,24,BLACK);
    tft.fillRect(0,60,128,15,BLACK);
    tft.fillRect(0,91,128,128,BLACK);
    Serial.println("End ");
    Serial.println(millis());
}


void oledArrow(float theta) {
    tft.setCursor(0, 0);

    battStat();
    sysStat();

    tft.drawLine(64,85,64+(40*cos((theta+90)*3.14/180)),85-(40*sin((theta+90)*3.14/180)),CYAN);
    tft.drawLine(64+(40*cos((theta+90)*3.14/180)),85-(40*sin((theta+90)*3.14/180)),(64+(30*cos((theta+100)*3.14/180))),(85-(30*sin((theta+100)*3.14/180))),CYAN);
    tft.drawLine(64+(40*cos((theta+90)*3.14/180)),85-(40*sin((theta+90)*3.14/180)),(64+(30*cos((theta+80)*3.14/180))),(85-(30*sin((theta+80)*3.14/180))),CYAN);
    tft.drawLine(64,85,64+(40*cos((theta+90)*3.14/180)),85-(40*sin((theta+90)*3.14/180)),BLACK);
    tft.drawLine(64+(40*cos((theta+90)*3.14/180)),85-(40*sin((theta+90)*3.14/180)),(64+(30*cos((theta+100)*3.14/180))),(85-(30*sin((theta+100)*3.14/180))),BLACK);
    tft.drawLine(64+(40*cos((theta+90)*3.14/180)),85-(40*sin((theta+90)*3.14/180)),(64+(30*cos((theta+80)*3.14/180))),(85-(30*sin((theta+80)*3.14/180))),BLACK);
}

void ledChangeHandler(uint8_t r, uint8_t g, uint8_t b)
{
    strip.setPixelColor(NETWORK_LED, r , g, b, 0); // Change this if your using regular RGB leds vs RGBW Leds.
    strip.show();

}

void obstacleAvoid(){
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

int getviconHeading(){
    struct command c;

    // Serial.println(millis());
    //Serial.println(millis());
    //Serial.println(bytes);

    // Loop while more than one packet is in buffer to clear it
    int bytes = client.available();
    while(bytes / sizeof(struct command) > 1) {
        // Read to clear old packets from buffer
        client.read((uint8_t *)(&c), sizeof(struct command));

        // Update bytes remainting in buffer
        bytes = client.available();
        Serial.print("x ");
    }
    bytes = client.read((uint8_t *)(&c), sizeof(struct command));

    //size_t bytes = client.read((uint8_t *)(&c), sizeof(struct command));
    // data parsing
    if(bytes > 0) {
        vicon_pos = (float)strtod(strtok(c.str, ","), NULL);
        vicon_theta = (float)strtod(strtok(NULL, ","), NULL);
    }
    // Serial.print("Recieve server update");
    // Serial.println(millis());
    // Serial.print("\tR: ");
    // Serial.print(pos);
    // Serial.print("\tTH: ");
    // Serial.print(theta);
    return bytes;
}

void getIMUHeading(){

    int t2 = millis();
    timeStep=t2-t1;
    float oldYaw=gz/131;
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    float yaw=gz/131;

            /* display Euler angles in degrees
            mpu.dmpGetQuaternion(&q, fifoBuffer);
            mpu.dmpGetGravity(&gravity, &q);
            mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
            Serial.print("yaw =");
            Serial.print(ypr[0]); //180/M_PI
            Serial.print("\t");
            //Serial.print(ypr[1] * 180/M_PI);
            //Serial.print("\t");
            //Serial.println(ypr[2] * 180/M_PI);
    */

      imu_theta=theta-((yaw+oldYaw)/2)*timeStep*.001;
      t1 = t2;

  // Wait to full timeStep period
  //delay((timeStep*1000) - (millis() - timer));

}
void headingUpdate(){
    getIMUHeading();
    if (getviconHeading() < 1) {
        Serial.println("imu heading");
         pos = imu_pos;
         theta = imu_theta;
    }
    else {
        Serial.println("Vicon Heading");
        pos = vicon_pos;
        theta = vicon_theta;
    }

}
