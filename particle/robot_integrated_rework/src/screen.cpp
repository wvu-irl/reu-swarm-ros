#include "screen.h"

Screen::Screen(void)
{
    //dummy
}

void Screen::init(void)
{
    oled.begin();
    oled.fillScreen(BLACK);
    oled.setTextSize(2);
    oled.setTextColor(WHITE);
    oled.println("Battery: ");
    oled.setTextSize(1);
    oled.setCursor(0, 28);
    oled.println("CONNECTIVITY: ");
    oled.drawTriangle(64, 38, 61, 43, 67, 43, WHITE); //Declares the front of the robot
}

void Screen::updateScreen(float _theta, bool _connected)
{

    sysStat(_connected);
    //battStat();
    oledArrow(_theta);
}

void Screen::sysStat(bool _connected)
{
    oled.setCursor(0, 0);
    if (_connected)
    {
        oled.fillCircle(83, 31, 4, GREEN);
    }
    else
    {
        oled.fillCircle(83, 31, 4, RED);
    }
}

void Screen::battStat()
{
    oled.setCursor(0, 18);
    oled.setTextSize(1);
    float voltage = analogRead(BATT) * 0.0011224;
    if (voltage >= 4.2) // If battery voltage is greater than 4.2 V it is at full charge
    {
        oled.setTextColor(GREEN);
        oled.println("Full Charge");
    }
    else if (voltage < 4.2 or voltage > 3.5) // If battery voltage is between 4.2 and 3.5 V battery is at safe usage levels
    {
        oled.setTextColor(YELLOW);
        oled.println("Useable Level");
    }
    else if (voltage <= 3.5) // If battery voltage is less than 3.5 V battery needs charged
    {
        oled.setTextColor(RED);
        oled.println("Needs Charging");
    }
}

void Screen::oledArrow(float _theta)
{
    oled.setCursor(0, 0);

    oled.drawLine(64, 85, 64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), CYAN);
    oled.drawLine(64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), (64 + (30 * cos((_theta + 100) * 3.14 / 180))), (85 - (30 * sin((_theta + 100) * 3.14 / 180))), CYAN);
    oled.drawLine(64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), (64 + (30 * cos((_theta + 80) * 3.14 / 180))), (85 - (30 * sin((_theta + 80) * 3.14 / 180))), CYAN);
    oled.drawLine(64, 85, 64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), BLACK);
    oled.drawLine(64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), (64 + (30 * cos((_theta + 100) * 3.14 / 180))), (85 - (30 * sin((_theta + 100) * 3.14 / 180))), BLACK);
    oled.drawLine(64 + (40 * cos((_theta + 90) * 3.14 / 180)), 85 - (40 * sin((_theta + 90) * 3.14 / 180)), (64 + (30 * cos((_theta + 80) * 3.14 / 180))), (85 - (30 * sin((_theta + 80) * 3.14 / 180))), BLACK);
}