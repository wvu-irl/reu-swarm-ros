#include <swarm_simulation/hawk_sim.h>




// ================ Callback functions ================================
void Hawk_Sim::chargersCallback(const wvu_swarm_std_msgs::chargers &msg)
{
	temp_chargers = msg;
}
void Hawk_Sim::priorityCallback(const wvu_swarm_std_msgs::priorities &msg)
{
	temp_priorities = msg;
}

void Hawk_Sim::energyCallback(const wvu_swarm_std_msgs::energy &msg)
{
	temp_energy = msg;
}

//=====================================================================

//==========================initializer functions=====================================================================
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
		if(temp_chargers.charger.size() > 0)
		{
			first = false;
		}
	}else
	{
		_pub.publish(temp_chargers);
	}
}
void Hawk_Sim::makePriority(ros::Publisher _pub)//creates chargers
{
	if (first)
	{
		wvu_swarm_std_msgs::priority priority_msg;
		priority_msg.priority = {0,10,2,3,4}; //{REST, CHARGE, CONTOUR, TARGET, EXPLORE}

		wvu_swarm_std_msgs::priorities priorities_msg;
		for(int i = 0; i < NUMBOTS; i ++)
		{
			priorities_msg.priorities.push_back(priority_msg);
		}
		_pub.publish(priorities_msg);
	}else
	{
		_pub.publish(temp_priorities);
	}
}
void Hawk_Sim::makeSensorData(ros::Publisher _pub)
{
	wvu_swarm_std_msgs::sensor_data sd_msg;

	sd_msg.rid = 1;
	_pub.publish(sd_msg);

}
void Hawk_Sim::makeEnergy(ros::Publisher _pub)
{
	if (energy_first)
	{
		wvu_swarm_std_msgs::energy energy_msg;
		for(int i = 0; i <NUMBOTS; i ++)
		{
			energy_msg.energies.push_back(100.0);
		}
		if(temp_chargers.charger.size() > 0)
		{
			energy_first = false;
		}
		_pub.publish(energy_msg);
	}else
	{
		_pub.publish(temp_energy);
	}
}
//========================================================================================================================

void Hawk_Sim::run(ros::NodeHandle n) // begin here
{
	NUMBOTS = 10;//set the number of bots.

	//Publishers
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::chargers > ("chargers", 1000); // pub to obstacles
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::priorities > ("priority", 1000); // pub to priority.
	ros::Publisher pub3 = n.advertise < wvu_swarm_std_msgs::energy > ("energy", 1000); // pub to energy.
	ros::Publisher pub4 = n.advertise < wvu_swarm_std_msgs::sensor_data > ("sensor_data", 1000); // pub to energy.

  // subscribers
	ros::Subscriber sub1 = n.subscribe("chargers", 1000, &Hawk_Sim::chargersCallback,this);
	ros::Subscriber sub2 = n.subscribe("priority", 1000, &Hawk_Sim::priorityCallback,this);
	ros::Subscriber sub3 = n.subscribe("energy", 1000, &Hawk_Sim::energyCallback,this);
	ros::Rate loopRate(10);

	int i = 0;
	while (ros::ok() && i < 10000) // setup loop
	{
		makeChargers(pub1);
		makePriority(pub2);
		makeEnergy(pub3);
		makeSensorData(pub4);
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
		makeChargers(pub1); // publishing chargers.
		makePriority(pub2); // publishes priority.
		makeEnergy(pub3);   // publishes energy.
		makeSensorData(pub4);//publishes sensor_data.
		ros::spinOnce(); // spinning callbacks
		loopRate.sleep();
	}
}
