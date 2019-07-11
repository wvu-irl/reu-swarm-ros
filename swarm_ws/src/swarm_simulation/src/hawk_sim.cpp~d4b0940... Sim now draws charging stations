#include <swarm_simulation/hawk_sim.h>




// ================ Callback functions ================================
void Hawk_Sim::chargersCallback(const wvu_swarm_std_msgs::chargers &msg)
{
	temp_chargers = msg;
}
//=====================================================================

//-------------initializer functions-----------------------------------
void Hawk_Sim::makeChargers(ros::Publisher _pub)//creates chargers
{
	if(first)
	{
		std::pair<float,float> a1 = {-50,0};
		std::pair<float,float> a2 = {-50,50};
		std::vector<std::pair<float,float>> coordinates = {a1,a2}; //positions of the charging stations

		wvu_swarm_std_msgs::chargers charger_vector;
		for (int i = 0; i < coordinates.size(); i++)
		{
			wvu_swarm_std_msgs::charger temp_charger;
			temp_charger.x = coordinates[i].first;
			temp_charger.y = coordinates[i].second;
			temp_charger.occupied = false;
			charger_vector.charger.push_back(temp_charger);
		}
		_pub.publish(charger_vector);
		if(temp_chargers.charger.size() > 0){first = false;}
	}else
	{
		_pub.publish(temp_chargers);
	}
}
//---------------------------------------------------------------------

void Hawk_Sim::run(ros::NodeHandle n) // begin here
{
	//Publisher
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::chargers > ("chargers", 1000); // pub to obstacles

  // subscribing
	ros::Subscriber sub1 = n.subscribe("chargers", 1000, &Hawk_Sim::chargersCallback,this);
	ros::Rate loopRate(10);

	makeChargers(pub1); //============changed for testing.=============
	int i = 0;
	while (ros::ok() && i < 10000) // setup loop
	{
		makeChargers(pub1);
		ros::spinOnce(); // spinning callbacks
//    usleep(10);
		i++; // incrementing counter
//
	}
	#if VOBJ_DEBUG
				 std::cout << "\033[35;1mStarting second loop\033[0m" << std::endl;
	#endif

	while (ros::ok()) // main loop
	{
		makeChargers(pub1); // publishing chargers
		ros::spinOnce(); // spinning callbacks
		loopRate.sleep();
	}
}
