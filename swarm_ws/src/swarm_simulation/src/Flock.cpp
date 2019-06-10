#include <swarm_simulation/Body.h>
#include <swarm_simulation/Flock.h>

//Flock Functions from Flock.h
//-----------------------------

int Flock::getSize()
{
    return flock.size();
}

Body Flock::getBody(int i)
{
    return flock[i];
}

wvu_swarm_std_msgs::vicon_bot_array Flock::createMessages() //generates an array of vicon_bot msgs.
{
	wvu_swarm_std_msgs::vicon_bot_array vb_array;
	for (int i = 0; i < flock.size();i++)
	{
		//initializes necessary variables for each iteration.
		wvu_swarm_std_msgs::vicon_bot this_bot;
		geometry_msgs::TransformStamped this_bot_msg;
		tf2::Quaternion q;
		Body cur = flock.at(i); //current body being looked at.

		float mag = cur.velocity.magnitude(); // r, the mag of the velocity
		q.setRPY( 0, 0, cur.angle(cur.velocity) - M_PI_2);  // Create this quaternion from roll=0/pitch=0/ yaw (in radians)
		//^will have to be changed for a holonomic (apparently direction and heading are different).
		q.normalize(); // normalizes the quaternion.



		//translational information
		this_bot_msg.transform.translation.x = cur.location.x * 0.07320644216; //scales info for table
		this_bot_msg.transform.translation.y = cur.location.y * 0.26041666666; //scales info for table
		this_bot_msg.transform.translation.z = 0;

		//rotational information
		this_bot_msg.transform.rotation.x = q.x();
		this_bot_msg.transform.rotation.y = q.y();
		this_bot_msg.transform.rotation.z = q.z();
		this_bot_msg.transform.rotation.w = q.w();

		//header information (dummy values)
		this_bot_msg.header.seq = 0;
		this_bot_msg.header.frame_id = "0";

		//set child frame dummy val.
		this_bot_msg.child_frame_id = "0";

		//create the vicon_bot.
		this_bot.botPose = this_bot_msg;
		this_bot.botId[0] = cur.id[0];
		this_bot.botId[1] = cur.id[1];

		//add to the vector list
		vb_array.poseVect.push_back(this_bot);
	}
	return vb_array;
}

void Flock::printMessage(wvu_swarm_std_msgs::vicon_bot_array _vb_array)
{//prints the frame_id, ID, and translational info of each bot's message.
	for (int i = 0; i < flock.size(); i ++)
	{
		std::cout<<"frame_id: "<<_vb_array.poseVect.at(i).botPose.header.frame_id<<std::endl;
		std::cout<<"x: "<<_vb_array.poseVect.at(i).botPose.transform.translation.x<<"\n";
		std::cout<<"y: "<<_vb_array.poseVect.at(i).botPose.transform.translation.y<<"\n";
		std::cout<<"z: "<<_vb_array.poseVect.at(i).botPose.transform.translation.z<<"\n";
		std::cout<<"ID: "<<_vb_array.poseVect.at(i).botId[0]<<_vb_array.poseVect.at(i).botId[1]<<"\n";
		std::cout<<"------------------------------------------\n";
	}
}


void Flock::addBody(Body b)
{
    flock.push_back(b);
}

// Runs the run function for every body in the flock checking against the flock
// itself. Which in turn applies all the rules to the flock.
void Flock::flocking()
{
    for (int i = 0; i < flock.size(); i++)
        flock[i].run(flock);
}
