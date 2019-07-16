#include "drivetrain.h"
#include "math.h"

DiffDrive::DiffDrive(void)
{
    servRight.attach(LEFTPIN);
    servLeft.attach(RIGHTPIN);
    lv = 3;
    lw = 2;

    error=0;
    oldError=0;
    dError=0;
    omega=0;
    kp=1;
    kd=0;
    aCommand=90;
    bCommand=-90;
    tStep=.01;
    tstep1=.001;
    start = true;
    l = .01;
}

void DiffDrive::init(void)
{
    //no-op for now?
}

void DiffDrive::drive(double _theta, double _speed, float yaw)
{
    // float v = lv * cos(_theta * 3.14 / 180);
    // float w = lw * sin(_theta * 3.14 / 180);

    if ((_theta >= 0 && _theta <= 90) || _theta > 270)
    {
        // servLeft.write(90 + 10 * (v - w));
        // servRight.write(90 - 10 * (v + w));
        unsigned long tstep2=millis();
        Serial.println(start);
        
        // l =yaw/((bCommand)-(aCommand));
        l=3.0;
        // if((bCommand-aCommand<.001)){
        //     l=.167;
        // }
#if DIFF_DRIVE_DEBUG
        Serial.printf("DIFF_DRIVE: \tl = %03.4f\n\r", l);
#endif
        error=_theta;
        if(_theta>180){
            error=_theta-360;
        }
        //error = atan2(sin(theta),cos(theta));
#if DIFF_DRIVE_DEBUG
        Serial.printf("DIFF_DRIVE: \tError = %03.4f\n\r", error);
#endif
        // Serial.print("Error");
        // Serial.println(error);
        dError=(error-oldError)/tStep;
#if DIFF_DRIVE_DEBUG
        Serial.printf("DIFF_DRIVE: \tdError = %03.4f\n\r", dError);
#endif
        // Serial.print("derror");
        // Serial.println(dError);
        omega=kp*error+kd*dError;
#if DIFF_DRIVE_DEBUG
        Serial.printf("DIFF_DRIVE: \tOmega = %03.4f\n\r", omega);
#endif
        // Serial.print("omega");
        // Serial.println(omega);
        
        if(_theta<180){
        // aCommand=-(omega*l)+bCommand;
            aCommand=-(omega*l)+90;
            bCommand=90;
            if(abs(aCommand)>90){
                aCommand= aCommand > 0.0 ? 90.0 : -90.0;
            }
        }
        else{
        // bCommand=(omega*l)+aCommand;
            bCommand=(omega*l)+90;
            aCommand=90;
            if(abs(bCommand)>90){
                bCommand= bCommand > 0.0 ? 90.0 : -90.0;
            }
            
        }
#if DIFF_DRIVE_DEBUG
        Serial.printf("DIFF_DRIVE: \taCommand = %03.4f\n\r\t\tbCommand = %03.4f\n\r", aCommand, bCommand);
#endif
        servLeft.write(aCommand+90); //180
        servRight.write(-bCommand+90); //0
        tStep=(double)(tstep2-tstep1)*.001;
#if DIFF_DRIVE_DEBUG
        Serial.printf("DIFF_DRIVE: \ttStep = %03.4f\n\r\t\ttStep1 = %lu\n\r\t\ttStep2 = %lu\n\r", tStep, tstep1, tstep2);
#endif
        tstep1=tstep2;
        start=false;
        oldError=error;
    }
    else if (_theta > 90 && _theta <= 180)
    { //we still want this because we want linear velocity to be non-negative ( change for pd)
        servLeft.write(90 - 10 * lw);
        servRight.write(90 - 10 * lw);
    }
    else if (_theta > 180 && _theta < 270)
    {
        servLeft.write(90 + 10 * lw);
        servRight.write(90 + 10 * lw);
    }
    else //if somehow the command is invalid
    {
        servLeft.write(90);
        servRight.write(90);
    }
}

// Completely disconnects servos to stop
void fullStop(void)
{
    servLeft.disconnect();
    servRight.disconnect();
}

// Reattaches servos. Call this after a fullStop
void restart(void)
{
    servRight.attach(LEFTPIN);
    servLeft.attach(RIGHTPIN);
}