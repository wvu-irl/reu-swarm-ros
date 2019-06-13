#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.cpp"
#include <sstream>

void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
}
void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
}
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg)
{
}

int main(int argc, char **argv)
{

	std::map<int, Robot::Robot> aliceMap;

	ros::init(argc, argv, "AliceBrain");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("target", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("virtual_obstacles", 1000, obsCallback);
	ros::Rate loopRate(10);
	ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::robot_command_array > ("final_execute", 1000);
	Hub aliceBrain(0); // Creates a hub for the conversion of absolute to relative info

	while (ros::ok())
	{

//	wvu_swarm_std_msgs::vicon_points tempTarget =
//						*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_points
//								> ("target"));
		wvu_swarm_std_msgs::vicon_bot_array tempBotArray = *(ros::topic::waitForMessage
				< wvu_swarm_std_msgs::vicon_bot_array > ("vicon_array"));
//        wvu_swarm_std_msgs::vicon_points temp_obs_array =
//        				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_points
//        						> ("virtual_obstacle"));

		aliceBrain.update(tempBotArray,tempTarget,temp_obs_array); //puts in absolute data from subscribers
		for (int i=0; i<tempBotArray.poseVect.size();i++){
			aliceMap[i].receiveMsg(aliceBrain.getAliceMail(i)); //gives each robot the relative data it needs
		}
		for (std::map<int, Robot::Robot>::iterator it=aliceMap.begin(); it!=aliceMap.end(); ++it) //eventually run this part asynchronously
		{
			//stuff
		}
		pub.publish(/*execute vector*/);
		ros::spinOnce();
		loopRate.sleep();
	}
	return 0;
}
