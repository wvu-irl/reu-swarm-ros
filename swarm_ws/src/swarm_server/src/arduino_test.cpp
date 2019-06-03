#include <ros/ros.h>
#include <wvu_swarm_std_msgs/robotcommand.h>
#include <wvu_swarm_std_msgs/sensor_data.h>

void sensorDatCallback(wvu_swarm_std_msgs::sensor_data msg)
{
	ROS_INFO("SENSOR DATA: %s", msg.data);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "arduino_test");
	ros::NodeHandle n;

	ros::Publisher exe = n.advertise< wvu_swarm_std_msgs::robotcommand >("execute", 1000);
	ros::Subscriber sense = n.subscribe("from_arduino", 1000, sensorDatCallback);

	ros::Rate loop_rate(100);

	while (ros::ok())
	{
		wvu_swarm_std_msgs::robotcommand cmd;
		cmd.rid = {'D', 'E'};
		cmd.r = 1.0f;
		cmd.theta = 123.4f;

		exe.publish(cmd);

		ros::spinOnce();
		loop_rate.sleep();
	}
}
