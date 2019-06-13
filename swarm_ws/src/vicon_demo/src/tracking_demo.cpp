#include <stdio.h>
#include <math.h>
#include <boost/array.hpp>
#include <ros/ros.h>
#include <ros/time.h>
#include <vicon_bridge/Markers.h>
#include <vicon_bridge/Marker.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <wvu_swarm_std_msgs/rtheta.h>
#include <wvu_swarm_std_msgs/vicon_bot.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>
#include <wvu_swarm_std_msgs/robot_command.h>
#include <wvu_swarm_std_msgs/robot_command_array.h>
#include "robot_id.h"

void msgCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg) {}

// Method to find distance between two points
double getDist(const geometry_msgs::Point first, const geometry_msgs::Point second);

// Method to populate a vector with a Gerono Lemniscate
//   Params: a vector to fill with the points, a value 'a' defining the curve,
//   and a value for an increment in degrees to iterate through 360 degrees
void genGeronoLemn(std::vector<geometry_msgs::Point> &target, const double a, const double incr);

void butterflyLemn(std::vector<geometry_msgs::Point> &target, const double a, const double incr);

// Method to process a bot's ideal vector
//   Params: a vector to add the bot's command to, the bot to use, the point to
//   compare the bot against.
//   Return: the distance of the bot
double processBot(wvu_swarm_std_msgs::robot_command_array &outputMsg, std::string tfPrefix,
        const wvu_swarm_std_msgs::vicon_bot bot, const geometry_msgs::Point point, tf2_ros::Buffer &tfBuff);

int main(int argc, char **argv)
{
    // Initialize a node
    ros::init(argc, argv, "tracker");
    
    // Generates nodehandle, publisher, subscribers
    ros::NodeHandle n;
    ros::NodeHandle n_priv("~"); // private handle
    ros::Publisher pub;
    ros::Subscriber sub;
    
    //  Where to get the array of bots, what topic to advertise, where to get transform of target
    std::string viconArrayTopic, advertiseTopic, transformPrefix;
    
    //  Value of 'a' in formula in cm, value to iterate degrees, distance to "find" points at
    int lemniscateConstant, lemniscateInterval, cutoffRadius;
    
    // Import parameters from launchfile
    n_priv.param<std::string>("vicon_array_topic", viconArrayTopic, "/viconArray");
    n_priv.param<std::string>("advertise_topic", advertiseTopic, "execute");
    n_priv.param<std::string>("transform_prefix", transformPrefix, "cardbot_");
    n_priv.param<int>("lemniscate_constant", lemniscateConstant, 50);
    n_priv.param<int>("lemniscate_interval", lemniscateInterval, 10);
    n_priv.param<int>("cutoff_radius", cutoffRadius, 10);
    
    // Sets loop rate at 100Hz
    ros::Rate rate(10);
    
    // Subscribe to tracker's vicon topic, advertise result vector
    sub = n.subscribe(viconArrayTopic, 10, &msgCallback);
    pub = n.advertise<wvu_swarm_std_msgs::robot_command_array>(advertiseTopic, 1000);
    
    // Set up transform buffer
    tf2_ros::Buffer tfBuffer;
    
    // Set up transform listener
    tf2_ros::TransformListener tfListener(tfBuffer);
    
    // Generate a Gerono Lemniscate (figure 8) as discrete points
    std::vector<geometry_msgs::Point> path;
    butterflyLemn(path, lemniscateConstant, lemniscateInterval);
    
    // Pick one point out of the path
    geometry_msgs::Point currentPoint = path.at(0);
    int currentIndex = 0;
    ROS_INFO("Point initialized at %f, %f\n", currentPoint.x, currentPoint.y);
    
    // Run while ros functions
    while(ros::ok())
    {
        // Get the information of the vicon array
        wvu_swarm_std_msgs::vicon_bot_array viconArray =
                *(ros::topic::waitForMessage<wvu_swarm_std_msgs::vicon_bot_array>(viconArrayTopic));
        
        // Generate a vector of robot commands to publish
        wvu_swarm_std_msgs::robot_command_array output;
        
        double minimumPointDistance = 10000.0; // Trivially large number, in cm
        
        // Iterate through all bots
        for(wvu_swarm_std_msgs::vicon_bot iteratorBot : viconArray.poseVect)
        {
            // Process bot's distance from the point and add its command vector to the output
            double thisDist = processBot(output, transformPrefix, iteratorBot, currentPoint, tfBuffer);
            
            // Change minimum if necessary
            if(thisDist < minimumPointDistance) minimumPointDistance = thisDist;
        }
        
        // Move the point if any bot has come too close
        if(minimumPointDistance < cutoffRadius) {
            // Rotate through vector if needed
            if(currentIndex + 1 >= path.size())
                currentIndex = 0;
            else
                currentIndex++;
            
            currentPoint = path.at(currentIndex);
        
            ROS_INFO("Point moved to %f, %f\n", currentPoint.x, currentPoint.y);
        }
        
        // Publish vector of robot commands
        pub.publish(output);
        
        ros::spinOnce();
        rate.sleep();
    }
}

double getDist(const geometry_msgs::Point first, const geometry_msgs::Point second)
{
    double xDiff = first.x - second.x;
    double yDiff = first.y - second.y;
    double zDiff = first.z - second.z;
    
    return sqrt(xDiff*xDiff + yDiff*yDiff + zDiff*zDiff);
}

void genGeronoLemn(std::vector<geometry_msgs::Point> &target, const double a, const double incr)
{
    // Iterate across 360 degrees
    for(int t = 0; t < 360; t += incr)
    {
        // Create a point
        geometry_msgs::Point temp;
        
        double rads = t * 3.14156265 / 180;
        
        // Do math to find its location
        //   This lemniscate is rotated 90 deg
        temp.x = a * cos(rads) * sin(rads);
        temp.y = a * sin(rads);
        temp.z = 0; //flatworld
        
        // Add to vector
        target.push_back(temp);
        
        ROS_INFO("Added point at %d degrees, or %f rads, at %f, %f, 0\n", t, rads, temp.x, temp.y);
    }
}

void butterflyLemn(std::vector<geometry_msgs::Point> &target, const double a, const double incr)
{
    // Iterate across 360 degrees
    for(int t = 0; t < 360; t += incr)
    {
        // Create a point
        geometry_msgs::Point temp;
        
        double rads = t * 3.14156265 / 180;
        
        // Do math to find its location
        //   This lemniscate is rotated 90 deg
        temp.x = a/3.0 * sin(rads) * (exp(cos(rads))-2*cos(4*rads)-(pow(sin(rads/12),5)));
        temp.y = a/3.0 * cos(rads) * (exp(cos(rads))-2*cos(4*rads)-(pow(sin(5/12),5)));
        temp.z = 0; //flatworld
        
        // Add to vector
        target.push_back(temp);
        
        ROS_INFO("Added point at %d degrees, or %f rads, at %f, %f, 0\n", t, rads, temp.x, temp.y);
    }
}

double processBot(wvu_swarm_std_msgs::robot_command_array &outputMsg, std::string tfPrefix,
        const wvu_swarm_std_msgs::vicon_bot bot, const geometry_msgs::Point point, tf2_ros::Buffer &tfBuff)
{
    // Generate a point for this bot's location
    geometry_msgs::Point botPt;
    botPt.x = bot.botPose.transform.translation.x;
    botPt.y = bot.botPose.transform.translation.y;
    botPt.z = 0;
    
    // Find distance between bot and point
    double dist = getDist(botPt, point);
    
    // Convert the bot's pose to a string
    char rid[3] = {'\0'};
    char rid2[3] = {'\0'};
    rid[0] = bot.botId[0];
    rid[1] = bot.botId[1];
    rid2[0] = bot.botId[0];
    rid2[1] = bot.botId[1];
    std::string idStr(rid);
    int idNum = rid_map[idStr];
    
    // Build a string for the robot's transformation
    std::string transformString = "vicon/";
    transformString += tfPrefix + idStr + "/" + tfPrefix + idStr;
    
    // Find transform from bot to target point
    geometry_msgs::TransformStamped trackerTarget;
    try{
        trackerTarget = tfBuff.lookupTransform(transformString,
                "world", ros::Time(0));
    } catch(tf2::LookupException) {
        ROS_ERROR("Can't find transform %s!", transformString.c_str());
        return 10000.0; // Arbitrarily large distance, cm
    }

    // Transform the point
    geometry_msgs::Point pointTransformed;
    tf2::doTransform<geometry_msgs::Point>(point, pointTransformed, trackerTarget);
    pointTransformed.z = 0; // why not

    // Find angle between bot and point
    double degrees = 0;
    if(pointTransformed.x < 0)
        degrees = atan(pointTransformed.y / pointTransformed.x) * 180 / 3.14159265;
    else
        degrees = (atan(pointTransformed.y / pointTransformed.x) * 180 / 3.14159265) + 180;
    if(degrees < 0) degrees += 360;
    
    // Build a command for this specific bot
    wvu_swarm_std_msgs::robot_command thisCmd;
    thisCmd.rid = idNum;
    thisCmd.r = dist;
    thisCmd.theta = degrees;
    
    // Add command to the vector
    outputMsg.commands.push_back(thisCmd);
    
    return dist;
}
