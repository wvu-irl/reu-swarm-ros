#include <stdio.h>
#include <math.h>
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

void trackCallback(const geometry_msgs::TransformStamped &msg) {}
void targetCallback(const geometry_msgs::TransformStamped &msg) {}

int main(int argc, char **argv)
{
    // Initialize a node
    ros::init(argc, argv, "vicon_demo");
    
    // Generates nodehandle, publisher, subscribers
    ros::NodeHandle n;
    ros::Publisher pub;
    ros::Subscriber trackerSub;
    ros::Subscriber targetSub;
    
    // Generate transform stuff
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer); // Receives tf2 transforms
    
    // Sets loop rate at 100Hz
    ros::Rate rate(100);
    
    // Subscribe to tracker's vicon topic, publish result vector
    trackerSub = n.subscribe("/vicon/tracker/tracker", 10, &trackCallback);
    targetSub = n.subscribe("/vicon/target/target", 10, &targetCallback);
    pub = n.advertise<vicon_demo::rtheta>("vicon_demo", 1000);
    //pub = n.advertise<geometry_msgs::Point>("vicon_demo", 1000);
    
    // Run while ros functions
    while(ros::ok())
    {
        // Get the transform of the tracker relative to origin
        geometry_msgs::TransformStamped trackerWorld =
                *(ros::topic::waitForMessage<geometry_msgs::TransformStamped>("/vicon/tracker/tracker"));
        
        // Get the transform of the target relative to origin
        geometry_msgs::TransformStamped targetWorld = 
                *(ros::topic::waitForMessage<geometry_msgs::TransformStamped>("/vicon/target/target"));
        
        // Find transform from tracker to target
        geometry_msgs::TransformStamped trackerTarget;
        trackerTarget = tfBuffer.lookupTransform("vicon/tracker/tracker",
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
        targetPtWorld.z = 0;
        
        // Publish the point of target relative to tracker
        //pub.publish(targetPtTracker);
        
        // Convert point to rtheta form
        float radius = sqrt(pow(targetPtTracker.x, 2) + pow(targetPtTracker.y, 2));
        radius = radius *= 50; //convert to cm
        
        float degrees = 0;
        
        // Make atan continuous across 0 to 360 deg
        if(targetPtTracker.x < 0)
            degrees = atan(targetPtTracker.y / targetPtTracker.x) * 180 / 3.14159265;
        else
            degrees = (atan(targetPtTracker.y / targetPtTracker.x) * 180 / 3.14159265) + 180;
        if(degrees < 0) degrees += 360;
        
        // Put into data structure
        vicon_demo::rtheta output;
        output.radius = radius;
        output.degrees = degrees;
        
        // Publish rtheta form
        pub.publish(output);
    }
}