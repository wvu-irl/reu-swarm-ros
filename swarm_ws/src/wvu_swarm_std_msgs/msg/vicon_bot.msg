# Robot ID in two-char form ("WV", "NC", "PA", etc)
uint8[2] botId

# Transform representing the robot's position relative to global frame
geometry_msgs/TransformStamped botPose

# identifiers
string name
bool selected