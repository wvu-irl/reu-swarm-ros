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
#include <wvu_swarm_std_msgs/vicon_bot.h>
#include <wvu_swarm_std_msgs/vicon_bot_array.h>

void msgCallback(const wvu_swarm_std_msgs::vicon_bot_array &msg) {}

int main(int argc, char **argv)
{
    // Initialize a node
    ros::init(argc, argv, "vicon_demo");
    
    // Generates nodehandle, publisher, subscribers
    ros::NodeHandle n;
    ros::Publisher pub;
    ros::Publisher visPub;
    ros::Subscriber sub;
    
    // Generate transform stuff
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer); // Receives tf2 transforms
    
    // Sets loop rate at 100Hz
    ros::Rate rate(100);
    
    // Subscribe to tracker's vicon topic, publish result vector
    sub = n.subscribe("/viconArray", 10, &msgCallback);
    pub = n.advertise<wvu_swarm_std_msgs::rtheta>("vicon_demo", 1000);
    visPub = n.advertise<visualization_msgs::MarkerArray>("demo_vis", 1000);
    //pub = n.advertise<geometry_msgs::Point>("vicon_demo", 1000);
    
    // Run while ros functions
    while(ros::ok())
    {
        // Get the information of the vicon array
        wvu_swarm_std_msgs::vicon_bot_array viconArray =
                *(ros::topic::waitForMessage<wvu_swarm_std_msgs::vicon_bot_array>("/viconArray"));
        
        // Look for two bots named AA and BB
        bool botAAFound = false, botBBFound = false;
        wvu_swarm_std_msgs::vicon_bot botAA, botBB;
        
        for(wvu_swarm_std_msgs::vicon_bot iteratorBot : viconArray.poseVect)
        {
            //std::uint8_t varA[2] = {(std::uint8_t)'A', (std::uint8_t)'A'};
            //std::uint8_t varB[2] = {(std::uint8_t)'B', (std::uint8_t)'B'};
            boost::array<std::uint8_t, 2> varA = {(std::uint8_t)'A', (std::uint8_t)'A'};
            boost::array<std::uint8_t, 2> varB = {(std::uint8_t)'B', (std::uint8_t)'B'};
            
            if(iteratorBot.botId == varA)
            {
                botAAFound = true;
                botAA = iteratorBot;
            }
            if(iteratorBot.botId == varB)
            {
                botBBFound = true;
                botBB = iteratorBot;
            }
        }
        
        // Iterate again if both not found
        if(!botAAFound || !botBBFound) {
            usleep(1000);
            continue;
        }
        
        // Pick out the geometry transforms from each bot relative to static frame
        //   Bot AA is our tracker, bot BB is our target
        geometry_msgs::TransformStamped trackerWorld = botAA.botPose;
        geometry_msgs::TransformStamped targetWorld = botBB.botPose;
        
        // Get the transform of the tracker relative to origin
        //geometry_msgs::TransformStamped trackerWorld =
        //        *(ros::topic::waitForMessage<geometry_msgs::TransformStamped>("/vicon/tracker/tracker"));
        
        // Get the transform of the target relative to origin
        //geometry_msgs::TransformStamped targetWorld = 
        //        *(ros::topic::waitForMessage<geometry_msgs::TransformStamped>("/vicon/target/target"));
        
        // Find transform from tracker to target
        geometry_msgs::TransformStamped trackerTarget;
        trackerTarget = tfBuffer.lookupTransform("vicon/swarmbot_AA/swarmbot_AA",
                "world", ros::Time(0));
        
        // Declare vectors for the target on the global frame and tracker frame
        geometry_msgs::Point targetPtWorld, targetPtTracker;
                
        // Grab the global frame point for the target
        targetPtWorld.x = targetWorld.transform.translation.x;
        targetPtWorld.y = targetWorld.transform.translation.y;
        targetPtWorld.z = targetWorld.transform.translation.z;
        
        // Transform target's global coordinates into target's tracker coordinates
        //   using the transform from global to tracker
        tf2::doTransform<geometry_msgs::Point>(targetPtWorld, targetPtTracker, trackerTarget);
        
        // Set z to zero because we're taking a trip to Flatland
        targetPtTracker.z = 0;
        
        // Publish the point of target relative to tracker
        //pub.publish(targetPtTracker);
        
        // Convert point to rtheta form
        float radius = sqrt(pow(targetPtTracker.x, 2) + pow(targetPtTracker.y, 2));
        
        float degrees = 0;
        
        // Make atan continuous across 0 to 360 deg
        if(targetPtTracker.x < 0)
            degrees = atan(targetPtTracker.y / targetPtTracker.x) * 180 / 3.14159265;
        else
            degrees = (atan(targetPtTracker.y / targetPtTracker.x) * 180 / 3.14159265) + 180;
        if(degrees < 0) degrees += 360;
        
        // Put into data structure
        wvu_swarm_std_msgs::rtheta output;
        output.radius = radius;
        output.degrees = degrees;
        
        // Publish rtheta form
        pub.publish(output);
        
        // Build some Markers for visualization
        visualization_msgs::Marker trackerMark, targetMark, trackerVectMark;
        
        trackerMark.header = trackerWorld.header;
        trackerMark.ns = "demo";
        trackerMark.id = 1;
        trackerMark.type = visualization_msgs::Marker::ARROW;
        trackerMark.pose.position.x = trackerWorld.transform.translation.x;
        trackerMark.pose.position.y = trackerWorld.transform.translation.y;
        trackerMark.pose.position.z = trackerWorld.transform.translation.z;
        trackerMark.pose.orientation.x = trackerWorld.transform.rotation.x;
        trackerMark.pose.orientation.y = trackerWorld.transform.rotation.y;
        trackerMark.pose.orientation.z = trackerWorld.transform.rotation.z;
        trackerMark.pose.orientation.w = trackerWorld.transform.rotation.w;
        trackerMark.scale.x = -0.2;
        trackerMark.scale.y = 0.04;
        trackerMark.scale.z = 0.04;
        trackerMark.color.r = 0;
        trackerMark.color.g = 1;
        trackerMark.color.b = 0;
        trackerMark.color.a = 1;
        
        targetMark.header = targetWorld.header;
        targetMark.ns = "demo";
        targetMark.id = 2;
        targetMark.type = visualization_msgs::Marker::SPHERE;
        targetMark.pose.position.x = targetWorld.transform.translation.x;
        targetMark.pose.position.y = targetWorld.transform.translation.y;
        targetMark.pose.position.z = targetWorld.transform.translation.z;
        targetMark.pose.orientation.x = targetWorld.transform.rotation.x;
        targetMark.pose.orientation.y = targetWorld.transform.rotation.y;
        targetMark.pose.orientation.z = targetWorld.transform.rotation.z;
        targetMark.pose.orientation.w = targetWorld.transform.rotation.w;
        targetMark.scale.x = 0.1;
        targetMark.scale.y = 0.1;
        targetMark.scale.z = 0.1;
        targetMark.color.r = 1;
        targetMark.color.g = 0;
        targetMark.color.b = 0;
        targetMark.color.a = 1;
        
        // Create a MarkerArray, publish it
        visualization_msgs::MarkerArray visOutput;
        visOutput.markers = {trackerMark, targetMark};
        visPub.publish(visOutput);
    }
}
