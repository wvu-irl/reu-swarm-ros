# Swarm Server Usage
[- Running from ROS](#running) 

[- Connecting from an arduino](#connecting)

[- Documentation](#documentation) 

## Running
To run the server the node ``ros_to_ardino_server`` must be started 



The robots are sent all the data from the topic ``execute``

All messages from the robots are able to be recieved in the topic ``from_ard``

## Connecting
To connect from an arduino 

- Arduino must connect to *IRL_ROBONET*
 
- Arduino must connect to relevant computer IP address

- Arduno must use port ``4321``

- To have the server communicate with the robot properly, the robot must ``register``

    - To register a robot the robot must send the command ``register <<RID>>`` where the RID is a numerical value from 0-49 

## Documentation

[Topics](#topics)

[File Specifics](#file-specifics)

[Example Code](#example-code)

#### Topics
Execution topic (``execute``)



- Uses message type ``robotcommand.msg`` to send messages
    - The message will change depending on swarm objectives
    - ``rid`` is the robot id represented as two ``uint8``'s containing a state abbreviation
    - ``r`` is the radius/velocity the roobot needs to meet
    - ``theta`` is the pose/angle the robot needs to meet in degrees  



- The robot specified in the RID in the message will be the only robot to recieve the message
    - special cases of "XX" and "YY" exist
        - "XX" sends a message to all robots 
      - "YY" sends a message to only logging devices
      

      
From arduino (``from_arduino``)



- Uses message type ``sensor_data`` to return sensor data
    - Currently uses a char[32] to return data **this will change** depending on sensors on robots
    
#### File Specifics

[ros_to_arduino_server.cpp](#ros_to_arduino_server-cpp)

[arduino_server.h & arduino_server_source.cpp](#arduino_server-h)

[robot_id.h](#robot_id-h)


This documentation is also attached to the source as well

---

##### `ros_to_arduino_server.cpp` 

**This is a node**

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

##### `arduino_server.h` 
##### & `arduino_server_source.cpp`

*NOTE: Only inlcude the header file*

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
        - `command_callback` is a callback function to process recieved data from the arduinos
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

##### `robot_id.h`

This header is here to convert from a numeric id to a `uint8[2]` or `string` id

feilds/globals:

- `rid_indexing` 
    - Is an array that contains all the state abbreviations in chronological order
    - this is meant to be used as a way to go from numeric to `string` ids

- `rid_map`
    - Is a map with `string` key types and `int` values
    - this is meant to convert from `string` ids to numeric ids

---
#### Example Code

This code is to be used when the `ros_to_arduino_server` node is also running

```
#include <ros/ros.h>

// messages usd by server 
#include <wvu_swarm_std_msgs/robotcommand.h>
#include <wvu_swarm_std_msgs/sensor_data.h>

// messages used by datastream
#include <wvu_swarm_std_msgs/rtheta.h>

/**
 * Callback for getting data from the sensor topic
 */
void sensorDatCallback(wvu_swarm_std_msgs::sensor_data msg)
{
	ROS_INFO("SENSOR DATA: %s", (char *) (&(msg.data[0])));
}

// main
int main(int argc, char **argv)
{
	// initializing the node
	ros::init(argc, argv, "server_test");
	ros::NodeHandle n;

	// advertizing the topic that the server subscribes to
	ros::Publisher exe = n.advertise < wvu_swarm_std_msgs::robotcommand
			> ("execute", 1000);
			
	// subscribing to the topic that the server produces
	ros::Subscriber sense = n.subscribe("from_arduino", 1000, sensorDatCallback);

	ros::Rate loop_rate(100); // setting a loop rate for sanity

	while (ros::ok())
	{
		/**
		 * This is just an example datastream
		 * Communication could work in a variety of other ways
		 */
		wvu_swarm_std_msgs::rtheta vector = *(ros::topic::waitForMessage
				< wvu_swarm_std_msgs::rtheta > ("/vicon_demo")); // getting ViCon data


		wvu_swarm_std_msgs::robotcommand cmd; // reparsing the message
		
		/**
		 *  if you wanted to change the input,
		 *  below is the place to do that
		 */
		
		cmd.rid =
		{ 'X', 'X'}; // setting the RID to send to all robots
		cmd.r = vector.radius; // getting vector information from datastream
		cmd.theta = vector.degrees;

		exe.publish(cmd); // sending the message to the robots

		ros::spinOnce(); // allowing callbacks to run
		loop_rate.sleep(); // sleeping for the proper loop rate
	}
}

```