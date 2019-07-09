#include <Hawk_Sim_Setup.h>




// ================ Callback functions ================================
void Hawk_Sim::chargersCallback(wvu_swarm_std_msgs::chargers &msg)
{
	temp_chargers = &msg;
}
//=====================================================================

//-------------initializer functions-----------------------------------
void Hawk_Sim::makeChargers(ros::Publisher _pub)//creates chargers
{
	std::vector<std::pair<float,float>> coordinates; //positions of the charging stations

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
}
//---------------------------------------------------------------------

void Hawk_Sim::run() // begin here
{
	//palce holders for subscription data ---------------
	wvu_swarm_std_msgs::chargers temp_chargers;
	//---------------------------------------------------

   // ros initialize
	ros::init(argc, argv, "Hawk_Sim_Setup");
	ros::NodeHandle n;
	ros::Publisher pub1 = n.advertise < wvu_swarm_std_msgs::vicon_points > ("chargers", 1000); // pub to obstacles

  // subscribing
	ros::Subscriber sub1 = n.subscribe("chargers", 1000, pointCallback);
	ros::Rate loopRate(100);
	sleep(2); //waits for sim to be awake

	int i = 0;
	while (ros::ok() && i < 10000) // setup loop
	{
		makeChargers(pub1);
		ros::spinOnce(); // spinning callbacks
    usleep(10);
		i++; // incrementing counter

	}
}
