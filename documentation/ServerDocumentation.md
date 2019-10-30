# Swarm Server Usage
[- Running from ROS](#running) 

[- Connecting from a device](#connecting)

[- Documentation](#documentation) 

## Running
To run the server the node `ros_to_ardino_server` must be started 



The robots are sent all the data from the topic ``execute``

All messages from the robots are able to be recieved in the topic ``from_ard``

## Connecting
To connect from a device 

- Device must connect to a valid network
    - A valid network must have a host computer running the TCP server
 
- Device must connect to relevant computer IP address

- Device must use port ``4321``

- To have the server communicate with the robot properly, the robot must ``register``

    - To register a robot the robot must send the command ``register <<RID>>`` where the RID is a 2 character state abbreviation

- To safely disconnect, the connected device should send `exit` to the server to appropriately disconnect and de-register.

## Documentation

[Topics](#topics)

[File Specifics](#file-specifics)

#### Topics
Execution topic (``execute``)

- Uses message type [robot_command_array.msg](MessageTypes.md#robot___command___array)

- Uses message type [robot_command.msg](MessageTypes.md#robot_command) to send messages


- The robot specified in the RID in the message will be the only robot to recieve the message
    - special cases of "XX" and "YY" exist
        - "XX" sends a message to all robots 
      - "YY" sends a message to only logging devices
      

      
From device (``from_arduino``)



- Uses message type [sensor_data](MessageTypes.md#sensor_data) to return sensor data
    - Currently uses a char[32] to return data **this will change** depending on sensors on robots
    
#### File Specifics

[ros_to_arduino_server.cpp](#ros_to_arduino_server-cpp)

[arduino_server.h & arduino_server_source.cpp](#arduino_server-h)

[robot_id.h](#robot_id-h)


This documentation is also attached to the source as well

---

##### ros_to_arduino_server.cpp 

**This is a node**

[Source code](https://github.com/wvu-irl/reu-swarm-ros/blob/master/swarm_ws/src/swarm_server/src/ros_to_arduino_server.cpp)

functions:

- ``flagger``:

    - This function is called when the program is interrupted by *ctrl+C*, this is what tells the program to shutdown when the interrupt happens
    - Modifies `g_flag`
    - Exiting is handeled in main loop

- ``commandCallback``
    - Callback for getting information from robots that is then packaged in ``sensor_data`` messages and published
    - publishes to `from_arduino` topic
    - The server was tested outside ROS in C++ only and having a callback precluded problems such as publishing
    
- ``info``
    - Callback for the server to print information about itself
    - The server was tested outside ROS in C++ only and having a callback precluded problems such as `ROS_INFO`
    
- ``sendToRobotCallback``
    - Subscription callback for `execute`
    - packages and sends data to swarm
    
- ``errorCallback``
    - Callback for the server to print error messages
    - The server was tested outside ROS in C++ only and having a callback precluded problems such as `ROS_ERROR`
    
- ``controlThread``
    - A function for a thread called in main to run the main loop of the node
    - Adds a signal catch to the program
    - subscribes to `execute`
    - begins loop and checking for exit conditions
    
- ``keepAlive``
    - A callback that tells the server it is ok to run
    - Contains a boolean expression that returns true if the server is to keep running
    
- ``main`` 
    - initializes ROS and creates the node
    - Creates a thread for the main control function
    - begins the server 
       - Server starts loop

globals:

- `g_from_ard`: Publisher for sensor data

- `g_flag`: Interrupt flag

---

##### arduino_server.h 
##### & arduino_server_source.cpp

*NOTE: Only inlcude the header file*

[Code for header](https://github.com/wvu-irl/reu-swarm-ros/blob/master/swarm_ws/src/swarm_server/src/arduino_server.h)

[Code for source](https://github.com/wvu-irl/reu-swarm-ros/blob/master/swarm_ws/src/swarm_server/src/arduino_server_source.cpp)

functions:

- `sendCommandToRobots`
    - Overloaded:
        - Passing in only a `command` will send a message to all connected robots
       - Otherwise a command and a valid two `char` string with state RID will send to a specific robot
    - Sends a command to robot(s)
    
- `runClient`
    - Runs in a thread created in `beginServer`
    - Acts as a "client handeler" to monitor the connection to the client to recieve messages
    - This function also handles registration of clients updating the appropriate `ConnectionInfo`

- `beginServer`
    - Begins the main server loop
    - Accepts callback functions for the printing of various information
        - `command_callback` is a callback function to process recieved data from the connected devices
        - `info_callback` is a callback to print information about server status
        - `error_callback` is a callback to print server errors 
        - `exit_condition_callback` is a callback that keeps the server running as long as it returns `true`

structs/classes:

- `struct command`
    - `typedef`ed struct for passing command information to the robots
    - contains only a `char[32]` to be parsed by the robot
    
- `class ConnectionInfo`
    - contains data related to the connection
    - `connection_descriptor` is a integer that corrisponds to the connection of the client
    - `rid` reffers to the robot id that is registered 
        - default is `-1` for the `rid`
        
- `struct client_param`
    - This struct is used to pass in the parameters of the client thread
    - Threads only allow `void *` inputs so a struct was used

---

##### robot_id.h

[Source code (in the main file)](https://github.com/wvu-irl/reu-swarm-ros/blob/master/swarm_ws/src/swarm_server/src/robot_id.h)

[Source code (in include)](https://github.com/wvu-irl/reu-swarm-ros/blob/master/swarm_ws/src/swarm_server/include/swarm_server/robot_id.h)

This header is here to convert from a numeric id to a `uint8[2]` or `string` id

**If you change the one in src do not forget the one in the include folder**

feilds/globals:

- `rid_indexing` 
    - Is an array that contains all the state abbreviations in chronological order
    - this is meant to be used as a way to go from numeric to `string` ids

- `rid_map`
    - Is a map with `string` key types and `int` values
    - this is meant to convert from `string` ids to numeric ids

---
##### funnel.cpp

[Source code](https://github.com/wvu-irl/reu-swarm-ros/blob/master/swarm_ws/src/swarm_server/src/funnel.cpp)

The purpose of this funnel is to create a collection of data that results from subscribing to a large number of topics.

functions:

- `comressionCallback`
    - A callback for the topic that was subscribed to in a thread
    - contains rules for a lock to keep away from write conflicts
        - The funnel uses a `std::vector` to collect data, which by nature is not thread safe.

- `listeningThread`
    - A thread started to subscribe to an individual topic.
    - This was done to allow a higher topic output rate.
    
- `main`
    - Creates the funnel node
    - starts threads so that they connect to the correct topics.
    - Looks for number of elements in the collected vector, and publishes them.
        - The topic published to is a collective of what is being subscribed to.