#include "ros/ros.h"
#include "std_msgs/String.h"
#include "Processor.cpp"
#include <sstream>

void botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
	//For Debug purposes. callback should do nothing by default.
	//ROS_INFO("I hear: [%i]", msg.id1);
}

int main(int argc, char **argv)
{

	// Test code for neighbor finding
	Bot a(0, 0, "ab");
	Bot b(1, 1, "bc");
	Bot c(2, 2, "cd");
	Bot d(3, 3, "de");
	Bot e(4, 4, "ef");
	Bot f(5, 5, "fg");
	Bot g(6, 6, "gh");
	Bot h(7, 7, "hi");
	Bot i(8, 8, "ij");
	Bot j(9, 9, "jk");
	Bot inputList[10] =
	{ a, b, c, d, e, f, g, h, i, j };

	//test code for obs finding
	std::pair<float, float> a1(1, 2);
	std::pair<float, float> a2(1, 3);
	std::pair<float, float> a3(1, 4);
	std::pair<float, float> a4(1, 5);
	// std::pair<float,float> a5 (2,1);
	// std::pair<float,float> a6 (2,7);
	// std::pair<float,float> a7 (2,3);
	// std::pair<float,float> a8 (2,4);
	// std::pair<float,float> a9 (2,5);
	// std::pair<float,float> a10 (-1,2);
	// std::pair<float,float> a11(-5,2);
	// std::pair<float,float> a12 (3,-2);
	// std::pair<float,float> a13 (-1,-2);
	// std::pair<float,float> a14 (-7,-8);
	std::pair<float, float> pair_array[4] =
	{ a1, a2, a3, a4 }; //,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14};

	Processor test_pros = Processor(inputList, pair_array);
	test_pros.findNeighbors();

	test_pros.printBotMail();
	wvu_swarm_std_msgs::alice_mail_array msg = test_pros.createAliceMsg(9);
	test_pros.printAliceMail(msg);
	std::cout << std::endl;

	Processor bigbrain(0);
	ros::init(argc, argv, "Processor");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("vicon_array", 1000, botCallback); //Subscribes to the Vicon
	std::vector < ros::Publisher > pubVector;

	for (int i = 0; i < BOT_COUNT; i++) //Starts publishing to all 50 topics
	{
		ros::Publisher pub = n.advertise < wvu_swarm_std_msgs::alice_mail_array
				> ("alice_mail_" + std::to_string(i), 1000);
		pubVector.push_back(pub);
	}

	while (ros::ok())
	{
		wvu_swarm_std_msgs::vicon_bot_array tempBotArray =
				*(ros::topic::waitForMessage < wvu_swarm_std_msgs::vicon_bot_array
						> ("vicon_array"));

		bigbrain.processVicon(tempBotArray);
		bigbrain.findNeighbors();

//     Bot a(0, 0, "ab");
//     Bot b(1, 1, "bc");
//     Bot c(2, 2, "cd");
//     Bot d(3, 3, "de");
//     Bot e(4, 4, "ef");
//     Bot f(5, 5, "fg");
//     Bot g(6, 6, "gh");
//     Bot h(7, 7, "hi");
//     Bot i(8, 8, "ij");
//     Bot j(9, 9, "jk");
//
//     std::pair<float,float> a1 (1,2);
//     std::pair<float,float> a2 (1,3);
//     std::pair<float,float> a3 (1,4);
//     std::pair<float,float> a4 (1,5);
//     /*std::pair<float,float> a5 (2,1);
//     std::pair<float,float> a6 (2,2);
//     std::pair<float,float> a7 (2,3);
//     std::pair<float,float> a8 (2,4);
//     std::pair<float,float> a9 (2,5);
//     std::pair<float,float> a10 (-1,2);
//     std::pair<float,float> a11(-5,2);
//     std::pair<float,float> a12 (3,-2);
//     std::pair<float,float> a13 (-1,-2);
//     std::pair<float,float> a14 (-7,-8);*/
//
//     std::pair<float,float> pair_array [4] = {a1,a2,a3,a4}; //,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14};
//     Bot inputList [10] = {a, b, c, d, e, f, g, h, i, j};
//
//     Processor test_pros = Processor(inputList, pair_array);
//     test_pros.findNeighbors();
//     std::cout << "x";
		for (int i = 0; i < BOT_COUNT; i++) //Publishes msgs to Alices
		{
			pubVector.at(i).publish(bigbrain.createAliceMsg(i));
		}
		ros::spinOnce();
	}
	return 0;
}
