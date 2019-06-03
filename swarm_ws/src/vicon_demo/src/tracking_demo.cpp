#include <stdio.h>
#include <math.h>
#include <boost/array.hpp>
#include <ros/ros.h>
#include <ros/time.h>
#include <vicon_demo/rtheta.h>
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
#include <wvu_swarm_std_msgs/viconBot.h>
#include <wvu_swarm_std_msgs/viconBotArray.h>
#include <wvu_swarm_std_msgs/typedPoint.h>

void msgCallback(const wvu_swarm_std_msgs::viconBotArray &msg) {}

// Method to find distance between two points
double getDist(const geometry_msgs::Point first, const geometry_msgs::Point second);

// Method to populate a vector with a Gerono Lemniscate
//   Params: a vector to fill with the points, a value 'a' defining the curve,
//   and a value for an increment in degrees to iterate through 360 degrees
void genGeronoLemn(std::vector<geometry_msgs::Point> &target, const double a, const double incr);

int main(int argc, char **argv)
{
    // Initialize a node
    ros::init(argc, argv, "vicon_demo");
    
    // Generates nodehandle, publisher, subscribers
    ros::NodeHandle n;
    ros::Publisher pub;
    ros::Subscriber sub;
    
    // Sets loop rate at 100Hz
    ros::Rate rate(100);
    
    // Subscribe to tracker's vicon topic, publish result vector
    sub = n.subscribe("/viconArray", 10, &msgCallback);
    pub = n.advertise<wvu_swarm_std_msgs::rtheta>("tracker_demo", 1000);
    
    // Set up transform listener
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);
    
    // Generate a Gerono Lemniscate (figure 8) as discrete points
    std::vector<geometry_msgs::Point> path;
    genGeronoLemn(path, 1, 10); // a = 1, increment 10 degrees per point
    
    // Pick one point out of the path
    geometry_msgs::Point currentPoint = path.at(0);
    int currentIndex = 0;
    ROS_INFO("Point initialized at %f, %f\n", currentPoint.x, currentPoint.y);
    
    // Run while ros functions
    while(ros::ok())
    {
        // Get the information of the vicon array
        wvu_swarm_std_msgs::viconBotArray viconArray =
                *(ros::topic::waitForMessage<wvu_swarm_std_msgs::viconBotArray>("/viconArray"));
        
        // Look for bot named AA
        bool botAAFound = false;
        wvu_swarm_std_msgs::viconBot botAA;
        
        for(wvu_swarm_std_msgs::viconBot iteratorBot : viconArray.poseVect)
        {
            boost::array<std::uint8_t, 2> varA = {(std::uint8_t)'A', (std::uint8_t)'A'};
            
            if(iteratorBot.botId == varA)
            {
                botAAFound = true;
                botAA = iteratorBot;
            }
        }
        
        // Iterate again if not found
        if(!botAAFound) {
            usleep(1000);
            continue;
        }
        
        // Pick out the point from bot relative to static frame
        geometry_msgs::Point trackerPt;
        trackerPt.x = botAA.botPose.transform.translation.x;
        trackerPt.y = botAA.botPose.transform.translation.y;
        trackerPt.z = 0;
        //ROS_INFO("   Tracker found at %f, %f\n", trackerPt.x, trackerPt.y);
        
        // Find distance between bot and its target, convert to cm
        double distance = getDist(trackerPt, currentPoint) * 50;
        
        // If less than 5cm, move point
        if(distance < 5) {
            // Rotate through vector if needed
            if(currentIndex + 1 >= path.size())
                currentIndex = 0;
            else
                currentIndex++;
            
            currentPoint = path.at(currentIndex);
            
            // Recalculate distance
            distance = getDist(trackerPt, currentPoint) * 50;
        
            ROS_INFO("Point moved to %f, %f\n", currentPoint.x, currentPoint.y);
        }
        
        // Find transform from tracker to target point
        geometry_msgs::TransformStamped trackerTarget;
        trackerTarget = tfBuffer.lookupTransform("vicon/swarmbot_AA/swarmbot_AA",
                "world", ros::Time(0));
        
        // Transform the point
        geometry_msgs::Point pointTransformed;
        tf2::doTransform<geometry_msgs::Point>(currentPoint, pointTransformed, botAA.botPose);
        pointTransformed.z = 0; // why not
        
        // Find angle between bot and point
        double degrees = 0;
        if(pointTransformed.x < 0)
            degrees = atan(pointTransformed.y / pointTransformed.x) * 180 / 3.14159265;
        else
            degrees = (atan(pointTransformed.y / pointTransformed.x) * 180 / 3.14159265) + 180;
        if(degrees < 0) degrees += 360;
        
        // Put into rtheta form, publish
        wvu_swarm_std_msgs::rtheta output;
        output.radius = distance;
        output.degrees = degrees;
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
        
        double rads = t * 180 / 3.14156265;
        
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