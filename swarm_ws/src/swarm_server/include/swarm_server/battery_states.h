/*
 * File:   battery_states.h
 * Author: Henry Vos
 *
 * Created on July 16, 2019, 4:39 PM
 */

#ifndef BATTERY_STATES_H
#define BATTERY_STATES_H

enum BATTERY_STATES
{
    ERROR = -2,
    NONE = -1,
    CHARGED,
    GOING,
    CHARGING
};

typedef enum BATTERY_STATES BatState;

#endif /* BATTERY_STATES_H */
