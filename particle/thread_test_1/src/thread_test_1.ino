/* TEST 1
 * QUESTION: Can while loops replace waitFor?
 * RESULTS: Setup prints after WiFi is connected, loop prints immediately after.
 *     Thus, while can replace waitFor. Further, loop() is not called until
 *     setup() has returned, even with threading on.
 */

SYSTEM_THREAD(ENABLED)

int count = 0;

void setup() {
  while(!WiFi.ready()) count++;
  Serial.begin(9600);
  Serial.print("setup: ");
  Serial.println(count); 
}

void loop() {
  Serial.print("loop: ");
  Serial.println(count);
  while(true); // hang here forever
}