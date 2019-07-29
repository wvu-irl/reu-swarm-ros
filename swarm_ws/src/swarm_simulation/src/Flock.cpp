#include <swarm_simulation/Body.h>
#include <swarm_simulation/Flock.h>

//Flock Functions from Flock.h
//-----------------------------

int Flock::getSize()
{
    return bodies.size();
}

Body Flock::getBody(int i)
{
    return bodies[i];
}

void Flock::addBody(Body b)
{
	bodies.push_back(b);
}

// Runs the run function for every body in the flock checking against the flock
// itself. Which in turn applies all the rules to the flock.
void Flock::applyPhysics(wvu_swarm_std_msgs::vicon_points *_targets)
{
    for (int i = 0; i < bodies.size(); i++)
    {
    	bodies[i].targets = _targets;
    	bodies[i].run(bodies);
    }
}

wvu_swarm_std_msgs::vicon_bot_array Flock::createMessages(
		wvu_swarm_std_msgs::vicon_bot_array real_bots) //generates an array of vicon_bot msgs.
{
	wvu_swarm_std_msgs::vicon_bot_array vb_array = real_bots;
	for (int i = 0; i < bodies.size();i++)
	{
		// filtering out real robots
		bool virt = true;
		do
		{
			virt = true;
			for (size_t j = 0; j < real_bots.poseVect.size(); j++)
			{
				if (real_bots.poseVect[j].botId[0] == bodies[i].id[0]
						&& real_bots.poseVect[j].botId[1] == bodies[i].id[1])
				{
					i++;
					virt = false;
				}
			}
		} while (!virt);
		if (i > bodies.size()) // safety
			break;

		//initializes necessary variables for each iteration.
		wvu_swarm_std_msgs::vicon_bot this_bot;
		geometry_msgs::TransformStamped this_bot_msg;
		tf2::Quaternion q;
		Body cur = bodies.at(i); //current body being looked at.

		float mag = cur.velocity.magnitude(); // r, the mag of the velocity

		//add negative sign to cur.angle ############################

		q.setRPY(0, 0, cur.heading); // Create this quaternion from roll=0/pitch=0/ yaw (in radians)
		q.normalize(); // normalizes the quaternion.

		//translational information
		this_bot_msg.transform.translation.x = cur.location.x * 0.3333 - 50; //scales info for table
		this_bot_msg.transform.translation.y = -cur.location.y * 0.3333 + 100; //scales info for table
		this_bot_msg.transform.translation.z = 0;

		//rotational information
		this_bot_msg.transform.rotation.x = q.x();
		this_bot_msg.transform.rotation.y = q.y();
		this_bot_msg.transform.rotation.z = q.z();
		this_bot_msg.transform.rotation.w = q.w();

		//header information (dummy values)
		this_bot_msg.header.seq = 0;
		this_bot_msg.header.frame_id = "0";
		this_bot_msg.header.stamp = ros::Time::now();
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
