#include "charger_status.h"



ChargerStatus::ChargerStatus(void)
{
    voltage = analogRead(BATT) * 0.0011224;
    pinMode(PWR, INPUT);
	pinMode(CHG, INPUT);
    // PWR: 0=no USB power, 1=USB powered     QI Charging is connected to USB
	// CHG: 0=charging, 1=not charging
    
   
    state = CHARGED;
}

String ChargerStatus::checkChargingState(void)
{
    String message = "";
    switch(state)
  {
    case CHARGED:
    //client.write this is my battery level
    if(voltage < 3.8){
        
        state = GOTOCHARGER;
        message="needs charging";
    }
    
    break;
       
    case GOTOCHARGER:
    //CLIENT PRINT GOING TO CHARGER
      if(voltage>=4){
        state = CHARGED;
        message="cancel charging ";
      }
      else if(PWR==1){
          state=CHARGING;
          message="charging! ";

      }
      break;
 
    case CHARGING:
      if(PWR==1 && CHG==0){
        //CLIENT PRINT CHARGING
        //CLIENT PRINT BATTERY LEVEL
      }
      else if(CHG==1 && voltage>4){
          state=CHARGED;
      }
      else if(PWR==0){
          state=GOTOCHARGER;
      }
      else{
          state=ERROR;
      }
      message="Charging-Battery Level:";
      break;
 
    case ERROR:
      if(CHG==1 && voltage>4){
          state=CHARGED;
      }
      else{
          //tone to signal error
          tone(A4,1760,1000);
      }
      break;
  }
  return message;
}
String ChargerStatus::giveBatteryVoltage(void)
{
    return String(voltage);
}

