# WVU Swarm Standard Messages
This file documents the ROS message types declared in our ``wvu_swarm_std_msgs`` package, located at ``swarm_ws/src/wvu_swarm_std_msgs``. These messages are designed to allow our ROS packages to communicate with each other in as standardized a way as possible.

## Maintenance
To add a message, follow the standard process of adding a ROS message. Simply add a file into ``wvu_swarm_std_msgs/msgs`` with the extension ``.msg`` with the variable types to be contained in that message. Then, be sure to add the name of your file to ``wvu_swarm_std_msgs/CMakeLists.txt``.

## Message Types
#### aliceMail
TODO
#### aliceMailArray
TODO
#### executeVector
TODO
#### robot_command
** The message will change depending on swarm objectives
    
- ``rid`` is the robot id represented as two ``uint8``'s containing a state abbreviation.
- ``r`` is the radius/velocity the roobot needs to meet.
- ``theta`` is the pose/angle the robot needs to meet in degrees. 
    
#### robot___command___array

Contains an array of `robot_command`s that `ros_to_arduino_server` will iterate through to send to the connected robots. 

- `commands` is an array of arbitrary size containing all the commands to be sent to the robot. 

#### rtheta
TODO
#### sensor_data
TODO
#### vicon_bot
TODO
#### vicon___bot___array
TODO
