# Robot ID in two-char form ("WV", "NC", "PA", etc)
# Robot ID of "XX" sends to all the robots
# Robot ID of "YY" sends to only logging computers
int32 rid

# Radius or velocity magnitude
float64 r

# Angle/Pose to be determined by robot
float64 theta 
#theta is negative in the counterclockwise direction.

string sys_comm