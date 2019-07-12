/* TEST 3
 * QUESTION: Does the OLED function properly from a thread?
 * RESULTS: OLED calls are still non-blocking when called from a thread.
 *     However, OLED speed has slowed significantly, suggesting that
 *     resources are handled differently (and slowly) in threads.
 */

SYSTEM_THREAD(ENABLED)

#include <Adafruit_mfGFX.h>
#include <Adafruit_SSD1351_Photon.h>

#define mosi D12 //blue - DIN - MO on Argon Board
#define sclk D13 //yellow
#define cs A5    //orange
#define dc D4    //green
#define rst D5   //white

#define PIXEL_PIN D7
#define PIXEL_COUNT 1
#define PIXEL_TYPE WS2812B

#define BLACK 0x0000
#define WHITE 0xFFFF

Adafruit_SSD1351 tft = Adafruit_SSD1351(cs, dc, mosi, sclk, rst);

Thread screenThread;
int milliCount = 0;

void screenLoop() {
  while(true) {
    tft.fillScreen(WHITE);
    Serial.print("WHITE: ");
    Serial.println(millis());
    tft.fillScreen(BLACK);
    Serial.print("BLACK: ");
    Serial.println(millis());
  }
}

void setup() {
  Serial.begin(9600);
  tft.begin();

  screenThread = Thread("screen", screenLoop);
}

void loop() {
  if(millis() - milliCount >= 1000) {
    Serial.println("1 second");
    milliCount = millis();
  }
  delayMicroseconds(10);
}