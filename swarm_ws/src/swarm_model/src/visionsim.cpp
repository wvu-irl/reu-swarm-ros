#include <stdio.h>
#include <cstdint>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <ros/master.h>
#include <swarm_model/Artifact.h> //items in a robot's vision
#include <swarm_model/Vision.h>   //one robot's vision
#include <swarm_model/Visions.h>  //container of all visions
#include <vicon_bridge/Markers.h>
#include <vicon_bridge/Marker.h>

void modelCallback(const swarm_model::Visions &msg)
{
    //TODO
}

int main(int argc, char **argv)
{
    // Initialize node
    ros::init(argc, argv, "model_vision_sim");
    ros::NodeHandle n,nh;
    
    // Create publisher
    //   Publishes an array with one entry for each robot containing
    //   what Artifacts it can see at what headings/distances.
    ros::Publisher visionPub = n.advertise<swarm_model::Visions>("model_visions", 1000);
    
    // Create subscriber
    ros::Subscriber viconSub;
    
    // 100Hz messages
    ros::Rate rate(100);
    
    // Run loop while ros is ok
    while(ros::ok()) {
        // Pull in the current array of Markers published by vicon_bridge
        // TODO: this
        
        // Declare lookup table, max id is 256
        // TODO: define max id in a launchfile
        std::vector<uint16_t> lookupTable;
        for(int i = 0; i < 256; i++) lookupTable.push_back(0);
        
        // Count those meeting needed naming scheme, add topic strings to a Vector
        ros::master::V_TopicInfo topicList;
        ros::master::getTopics(topicList);
        std::vector<std::string> topicStrings = {};
        
        int robotCount = 0;
        
        // For each topic in list, check if it's our bot
        for(ros::master::TopicInfo currTopic : topicList)
        {
            if(currTopic.name.find("swarmbot_") == std::string::npos) // TODO: parametrize name in launchfile
            {
                // Add topic to list
                topicStrings.push_back(currTopic.name);
                robotCount++;
            }
        }
        
        // Declare an array to store Markers
        vicon_bridge::Marker robotMarkers[robotCount];
        
        // Iterate through topics, getting Markers
        for(std::string currentTopic : topicStrings)
        {
            viconSub = n.subscribe(currentTopic, 1000, modelCallback);
        }
        
        // Declare robot visions to publish
        swarm_model::Visions message;
        
        // Declare Vectors to construct Visions arrays
        std::vector<swarm_model::Artifact> currentArtifacts = {};
        std::vector<swarm_model::Vision> allRobots = {};
        
        // For each Marker, calculate distances to all other Markers
        for(int i = 0; i < robotCount; i++)
        {
            vicon_bridge::Marker currentMarker = robotMarkers[i];
            
            // Iterate through all other robot instances, find distances
            for(int j = 0; j < robotCount; j++)
            {
                // Ignore if same robot
                if(i == j) continue;
                
                vicon_bridge::Marker otherMarker = robotMarkers[j];
                
                float distance = 1; //TODO: Math
                
                // Add artifact to vision vector if in range
                if(distance < 1) //TODO: Make distance a launchfile parameter
                {
                    // Build an Artifact from the current Marker
                    swarm_model::Artifact thisArtifact;
                    // TODO: this
                    
                    // Add the Artifact to the list
                    currentArtifacts.push_back(thisArtifact);
                }
            } // End iterating through other robots
                
            // Find id, add to lookup table
            int idIndex = currentMarker.subject_name.find("_") + 1;
            int id = std::stoi(currentMarker.subject_name.substr(idIndex));
            lookupTable[id] = i;
            
            // Construct a Vision for this robot
            swarm_model::Vision thisRobot;
            thisRobot.id = id;
            thisRobot.count = currentArtifacts.size();
            thisRobot.sight = currentArtifacts;
            
            // Add this robot to the total list
            allRobots.push_back(thisRobot);
            
            // Reset the Artifacts for the next robot
            currentArtifacts.clear();
        } // End iterating through robots
            
        // Write to message
        message.count = allRobots.size();
        message.lookup = lookupTable;
        message.model = allRobots;
        
        // Publish the visions
        visionPub.publish(message);
        
        // Clear vectors
        lookupTable.clear();
        allRobots.clear();
        
        // Sleep and spin to collect messages
        rate.sleep();
        ros::spinOnce();
    }
    
    rate.sleep();
    ros::spinOnce();
}