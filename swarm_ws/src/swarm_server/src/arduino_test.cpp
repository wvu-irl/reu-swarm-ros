#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robotcommand.h>
#include <wvu_swarm_std_msgs/sensor_data.h>
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
	ros::init(argc, argv, "arduino_test");
	ros::NodeHandle n;

	// acting as publisher
	ros::Publisher exe = n.advertise < wvu_swarm_std_msgs::robotcommand
			> ("execute", 1000); // creating topic for the server to listen to
	ros::Subscriber sense = n.subscribe("from_arduino", 1000, sensorDatCallback);

	sleep(10);

	ros::Rate loop_rate(100);

	while (ros::ok())
	{
		wvu_swarm_std_msgs::rtheta vector = *(ros::topic::waitForMessage
				< wvu_swarm_std_msgs::rtheta > ("/vicon_demo")); // getting ViCon data

		wvu_swarm_std_msgs::robotcommand cmd; // reparsing the message
		cmd.rid =
		{ 'X', 'X'};
		cmd.r = vector.radius;
		cmd.theta = vector.degrees;

		exe.publish(cmd); // sending the message

		ros::spinOnce();
		loop_rate.sleep();
	}
}
