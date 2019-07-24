#include <swarm_simulation/hawk_sim.h>

// ================ Callback functions ================================
void Hawk_Sim::chargersCallback(const wvu_swarm_std_msgs::chargers &msg)
{

	temp_chargers = msg;
	if (first)
	{
		prev_temp_chargers = temp_chargers;
	}
	new_chargers = false;
	for (int i = 0; i < prev_temp_chargers.charger.size(); i++)
	{
		if (temp_chargers.charger.at(i).occupied != prev_temp_chargers.charger.at(i).occupied)
		{
			new_chargers = true;
		}
	}
	if (new_chargers)
	{
		prev_temp_chargers = temp_chargers;
	} else
	{
		temp_chargers = prev_temp_chargers;
	}

}
void Hawk_Sim::priorityCallback(const wvu_swarm_std_msgs::priorities &msg)
{
	temp_priorities = msg;
}

void Hawk_Sim::energyCallback(const wvu_swarm_std_msgs::energy &msg)
{
	temp_energy = msg;
}

void Hawk_Sim::botCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg)
{
	temp_bots = msg;
}

//=====================================================================

//==========================initializer functions=====================================================================
void Hawk_Sim::makeChargers(ros::Publisher _pub) //creates chargers
{
	if (first)
	{
		std::pair<float, float> a1 =
		{ -50, 0 };
		std::pair<float, float> a2 =
		{ -50, 50 };
		std::vector<std::pair<float, float>> coordinates =
		{ a1, a2 }; //positions of the charging stations

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
		if (temp_chargers.charger.size() > 0)
		{
			first = false;
			prev_temp_chargers = temp_chargers;
		}
	} else
	{
//		if(new_chargers)
//		{}
		_pub.publish(temp_chargers);
	}
}
void Hawk_Sim::makePriority(ros::Publisher _pub) //creates chargers
{
	if (first)
	{
		wvu_swarm_std_msgs::priority priority_msg;
		priority_msg.priority =
		{ 0,10,2,3,4}; //{REST, CHARGE, CONTOUR, TARGET, EXPLORE}

		wvu_swarm_std_msgs::priorities priorities_msg;
		for (int i = 0; i < NUMBOTS; i++)
		{
			priorities_msg.priorities.push_back(priority_msg);
		}
		_pub.publish(priorities_msg);
	} else
	{
		_pub.publish(temp_priorities);
	}
}
void Hawk_Sim::makeSensorData(ros::Publisher _pub)
{
	if (prev_rid < 0)
	{
		prev_rid = 0;
	}
	wvu_swarm_std_msgs::sensor_data sd_msg;
	sd_msg.rid = prev_rid;
	sd_msg.battery_level = 2;
	sd_msg.battery_state = GOING;


	if(counter > 1080 && counter < 1150)
	{
		sd_msg.battery_state = CHARGING;
		sd_msg.battery_level = 2;
	}
	else if(counter >= 1150)
	{
		sd_msg.battery_level = 5;
		sd_msg.battery_state = CHARGED;
	}
	counter += 1;
//	std::cout<<"counter "<<counter<<std::endl;
	_pub.publish(sd_msg);

	int i = 0;
	int x = 3;
	while (i < x)
	{
		if (prev_rid == i)
		{
			prev_rid = prev_rid + 1;
			i = NUMBOTS;
			if (prev_rid == x)
			{
				prev_rid = 0;
			}
		}
		i++;
	}
}
void Hawk_Sim::makeEnergy(ros::Publisher _pub)
{
	//I think what you tried here is to send out the energy once and have it loop afterwards, but energy doesn't seem like the thing the robot's decide for themselves
//	if (energy_first)
//	{
//		wvu_swarm_std_msgs::energy energy_msg;
//		for(int i = 0; i <NUMBOTS; i ++)
//		{
//			energy_msg.energies.push_back(100.0);
//		}
//		if(temp_chargers.charger.size() > 0)
//		{
//			energy_first = false;
//		}
//		_pub.publish(energy_msg);
//	}else
//	{
//		_pub.publish(temp_energy);
//	}

}
void Hawk_Sim::makeTargets(ros::Publisher _pub)
{

	wvu_swarm_std_msgs::vicon_points target_msg;
	std::vector<std::pair<float, wvu_swarm_std_msgs::vicon_point>>::iterator food_it = food_targets.begin();
	float tstep = (float) (ros::Time::now().toSec() - time.toSec()); //the time elapsed since the last run
	time = ros::Time::now(); //sets time here

	while (food_it != food_targets.end())
	{ //iterate through each food point
		for (int i = 0; i < temp_bots.poseVect.size(); i++)
		{ //iterate through each robot
			if (pow(
					pow(temp_bots.poseVect.at(i).botPose.transform.translation.x - food_it->second.x, 2)
							+ pow(temp_bots.poseVect.at(i).botPose.transform.translation.y - food_it->second.y, 2), 0.5) < 5) //check for eating distance
			{
				food_it->first -= tstep *1; //sets the rate of food consumption.
			}
		}
		if (food_it->first < 0)
			food_it = food_targets.erase(food_it); //erase the food if it is fully consumed.
		else
			food_it++;
	}
	if (food_targets.size() < 4)
	{ //create another location with food if there are too few
		wvu_swarm_std_msgs::vicon_point temp1;
		temp1.x = rand() % 80 - 40; //randomize the location.
		temp1.y = rand() % 160 - 80;
		std::pair<float, wvu_swarm_std_msgs::vicon_point> temp2(1, temp1); //start with a quantity of 1
		food_targets.push_back(temp2);
	}
	for (int i = 0; i < food_targets.size(); i++)
	{
		target_msg.point.push_back(food_targets.at(i).second); //add the targets to the msg

	}
	_pub.publish(target_msg);

}
//========================================================================================================================

void Hawk_Sim::run(ros::NodeHandle n) // begin here
{
//==============Enter the number of bots desired here=================
	NUMBOTS = 10;
//====================================================================

	//----------------------------Publishers-------------------------------------
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::chargers > ("chargers", 1000); // pub to obstacles
	ros::Publisher pub2 = n.advertise < wvu_swarm_std_msgs::priorities > ("priority", 1000); // pub to priority.
	//ros::Publisher pub3 = n.advertise < wvu_swarm_std_msgs::energy > ("energy", 1000); // pub to energy.
	//Ideally energy would come from here. However I have chosen the practical+nonideal implementation of simply
	//keeping energy contained and updated within each robot. No passing to worry about this way.
	ros::Publisher pub4 = n.advertise < wvu_swarm_std_msgs::sensor_data > ("sensor_data", 1000); // pub to energy.
	ros::Publisher pub5 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("virtual_targets", 1000);
	// ------------------------------Subscribers----------------------------------------
	ros::Subscriber sub1 = n.subscribe("chargers", 1000, &Hawk_Sim::chargersCallback, this);
	ros::Subscriber sub2 = n.subscribe("priority", 1000, &Hawk_Sim::priorityCallback, this);
	ros::Subscriber sub3 = n.subscribe("energy", 1000, &Hawk_Sim::energyCallback, this);
	ros::Subscriber sub4 = n.subscribe("vicon_array", 1000, &Hawk_Sim::botCallback, this);
	ros::Rate loopRate(10);

	int i = 0;
	while (ros::ok() && i < 1000) // setup loop
	{
		makeChargers(pub1);
		makePriority(pub2);
//		makeEnergy(pub3);
		makeSensorData(pub4);
		makeTargets(pub5);
		ros::spinOnce(); // spinning callbacks
		i++; // incrementing counter

	}
#if VOBJ_DEBUG
				 std::cout << "\033[35;1mStarting second loop\033[0m" << std::endl;
	#endif

	while (ros::ok()) // main loop
	{
		makeChargers(pub1); // publishing chargers.
		makePriority(pub2); // publishes priority.
		//makeEnergy(pub3);   // publishes energy.
		makeSensorData(pub4);		//publishes sensor_data.
		makeTargets(pub5);
		ros::spinOnce(); // spinning callbacks
		loopRate.sleep();
	}
}
