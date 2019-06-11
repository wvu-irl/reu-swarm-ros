#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.cpp"
#include <sstream>

void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
	//For Debug purposes. callback should do nothing by default.
	//ROS_INFO("I hear: [%i]", msg.id1);
}
void obsCallback(const wvu_swarm_std_msgs::vicon_points &msg){}
void pointCallback(const wvu_swarm_std_msgs::vicon_points &msg){}

int main(int argc, char **argv)
{
//	// Test code for neighbor finding
//		Bot a(0, 0, 0);
//		Bot b(1, 1, 1);
//		Bot c(2, 2, 2);
//		Bot d(3, 3, 3);
//		Bot e(4, 4, 4);
//		Bot f(5, 5, 5);
//		Bot g(6, 6, 6);
//		Bot h(7, 7, 7);
//		Bot i(8, 8, 8);
//		Bot j(9, 9, 9);
//		Bot inputList[10] =
//		{ a, b, c, d, e, f, g, h, i, j };
//
//		//test code for obs finding
//		std::pair<float, float> a1(1, 2);
//		std::pair<float, float> a2(1, 3);
//		std::pair<float, float> a3(1, 4);
//		std::pair<float, float> a4(1, 5);
//		std::pair<float, float> pair_array[4] =
//		{ a1, a2, a3, a4 };
//
//		Processor test_pros = Processor(inputList, pair_array);
//		test_pros.findNeighbors();
//
//		test_pros.printBotMail();
//		wvu_swarm_std_msgs::alice_mail_array msg = test_pros.createAliceMsg(9);
//		test_pros.printAliceMail(msg);
//		std::cout << std::endl;

	Processor bigbrain(0);

	ros::init(argc, argv, "Processor");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	ros::Subscriber sub2 = n.subscribe("target", 1000, pointCallback);
	ros::Subscriber sub3 = n.subscribe("virutal_obstacles",1000,obsCallback);
	std::vector < ros::Publisher > pubVector;

	for (int i = 0; i < BOT_COUNT; i++) //Starts publishing to all 50 topics
	{

		ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::alice_mail_array
				> ("alice_mail_" + std::to_string(i), 1000);
		pubVector.push_back(pub);
	}

	while (ros::ok())
	{

//	wvu_swarm_std_msgs::vicon_points tempTarget =
//						*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_points
//								> ("target"));
       
        wvu_swarm_std_msgs::vicon_bot_array tempBotArray =
				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_bot_array
						> ("vicon_array"));

        wvu_swarm_std_msgs::vicon_points temp_obs_array =
        				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_points
        						> ("virtual_obstacle"));
//
//		bigbrain.processPoints(tempTarget);
		bigbrain.processVicon(tempBotArray);
		bigbrain.processObstacles(temp_obs_array);
		bigbrain.findNeighbors();
                for (int i = 0; i < BOT_COUNT; i++) //Publishes msgs to Alices
		{
                    
			pubVector.at(i).publish(bigbrain.createAliceMsg(i));
		}
                bigbrain.clearProcessor();
		ros::spinOnce();
	}
	return 0;
}
