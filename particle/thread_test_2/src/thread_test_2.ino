/* TEST 2
 * QUESTION: Are OLED functions blocking?
 * RESULTS: OLED slowly cycles between white and black. Serial outputs
 *     update at same, slow rate. Thus, OLED commands are blocking.
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

void setup() {
  Serial.begin(9600);
  tft.begin();
}

void loop() {
  tft.fillScreen(WHITE);
  Serial.print("WHITE: ");
  Serial.println(millis());
  tft.fillScreen(BLACK);
  Serial.print("BLACK: ");
  Serial.println(millis());
}