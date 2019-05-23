std_msgs/Header header

# marker id
int16 id

# 3D space
geometry_msgs/Vector3 rvec
geometry_msgs/Vector3 tvec

# roll, pitch, yaw
geometry_msgs/Vector3 rpy

# pixel coordinates of corners (theta from Pose2D not used)
geometry_msgs/Pose2D[] pixel_corners
