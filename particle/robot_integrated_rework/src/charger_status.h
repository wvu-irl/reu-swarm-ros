#ifndef CHARGER_STATUS_H
#define CHARGER_STATUS_H

#include <Particle.h>
#include <string.h>

class ChargerStatus
{
public:
    // Public Methods & variables
    ChargerStatus(void);
    String checkChargingState(void);
    String giveBatteryVoltage(void);

private:
    //Private variables
    float voltage;
    uint8_t state;
    enum State_enum {CHARGED, GOTOCHARGER,CHARGING,ERROR};
    String message;


};

#endif