#include "screen.h"

Screen::Screen(void)
{
    //dummy
}

void Screen::init(String _reg)
{
    oled.begin();
    oled.fillScreen(BLACK);
    oled.setCursor(97, 0);
    oled.setTextSize(1);
    oled.setFont(TIMESNR_8);
    oled.print(_reg);
}

void Screen::updateScreen(float _theta, bool _connected)
{
    sysStat(_connected);
    battStat();
    oledArrow(_theta);
}

void Screen::sysStat(bool _connected)
{
    if(_connected != tcpConnected) {
        tcpConnected = _connected;
        if (_connected)
        {
            oled.fillCircle(123, 4, 4, GREEN);
        }
        else
        {
            oled.fillCircle(123, 4, 4, RED);
        }
    }
}

void Screen::battStat()
{
    // Check voltage
    enum BATT_STATE currentState;
    float voltage = analogRead(BATT) * 0.0011224;
    // TODO: check for charging
    if (voltage >= 4.15) // If battery voltage is greater than 4.15V it is close enough to charged-- Particle won't allow pin to pass ~4.19
    {
        currentState = B_HIGH;
    }
    else if (voltage > 3.5) // If battery voltage is between 4.15 and 3.5 V battery is at safe usage levels
    {
        currentState = B_MED;
    }
    else // If battery voltage is less than 3.5 V battery needs charged
    {
        currentState = B_LOW;
    }

    // Change value of battery and update screen if value has changed
    if(currentState != batteryState)
    {
        batteryState = currentState;

        if(batteryState == B_HIGH)
        {
            oled.fillRect(0, 0, 19, 9, GREEN);
            oled.drawRect(1, 1, 17, 7, BLACK);
            oled.drawRect(20, 2, 2, 5, GREEN);
        }
        else if(batteryState == B_MED)
        {
            oled.fillRect(0, 0, 19, 9, YELLOW);
            oled.drawRect(1, 1, 17, 7, BLACK);
            oled.fillRect(13, 2, 5, 5, BLACK);
            oled.drawRect(20, 2, 2, 5, YELLOW);
        }
        else if(batteryState == B_LOW)
        {
            oled.fillRect(0, 0, 19, 9, RED);
            oled.drawRect(1, 1, 17, 7, BLACK);
            oled.fillRect(8, 2, 10, 5, BLACK);
            oled.drawRect(20, 2, 2, 5, RED);
        }
        else   // charging
        {
            // TODO: lightning or something
            oled.fillRect(0, 0, 19, 9, YELLOW);
            oled.drawRect(1, 1, 17, 7, BLACK);
            oled.fillRect(13, 2, 5, 5, BLACK);
            oled.drawRect(20, 2, 2, 5, YELLOW);
        }
    }
}

void Screen::oledArrow(float _theta)
{
    // Only update if theta has changed
    if(_theta != oldTheta) {
        // Erase old arrow
        oled.drawLine(64, 85, 64 + (40 * cos((oldTheta + 90) * 3.14 / 180)), 85 - (40 * sin((oldTheta + 90) * 3.14 / 180)), BLACK);
        oled.drawLine(64 + (40 * cos((oldTheta + 90) * 3.14 / 180)), 85 - (40 * sin((oldTheta + 90) * 3.14 / 180)), (64 + (30 * cos((oldTheta + 100) * 3.14 / 180))), (85 - (30 * sin((oldTheta + 100) * 3.14 / 180))), BLACK);
        oled.drawLine(64 + (40 * cos((oldTheta + 90) * 3.14 / 180)), 85 - (40 * sin((oldTheta + 90) * 3.14 / 180)), (64 + (30 * cos((oldTheta + 80) * 3.14 / 180))), (85 - (30 * sin((oldTheta + 80) * 3.14 / 180))), BLACK);

        // Draw new arrow
        oled.drawLine(64, 85, 64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), CYAN);
        oled.drawLine(64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), (64 + (30 * cos((_theta + 100) * 3.14 / 180))), (85 - (30 * sin((_theta + 100) * 3.14 / 180))), CYAN);
        oled.drawLine(64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), (64 + (30 * cos((_theta + 80) * 3.14 / 180))), (85 - (30 * sin((_theta + 80) * 3.14 / 180))), CYAN);

        // Update old theta
        oldTheta = _theta;
    }
}