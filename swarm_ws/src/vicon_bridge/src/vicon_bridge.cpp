/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, UC Regents
 *  Copyright (c) 2011, Markus Achtelik, ETH Zurich, Autonomous Systems Lab (modifications)
 *  Copyright (c) 2019, WVU Interactive Robotics Lab (modifications)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the University of California nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include <Client.h>
#include <ros/ros.h>
#include <diagnostic_updater/diagnostic_updater.h>
#include <diagnostic_updater/update_functions.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <wvu_swarm_std_msgs/viconBot.h>
#include <wvu_swarm_std_msgs/viconBotArray.h>
#include <vicon_bridge/viconGrabPose.h>
#include <iostream>
#include <string>
#include <cstdint>

#include <vicon_bridge/Markers.h>
#include <vicon_bridge/Marker.h>

#include "msvc_bridge.h"
#include <map>
#include <boost/thread.hpp>
#include <vicon_bridge/viconCalibrateSegment.h>
#include <tf/transform_listener.h>

using std::min;
using std::max;
using std::string;
using std::map;

using namespace ViconDataStreamSDK::CPP;

// Puts Direction constants into readable format
string Adapt(const Direction::Enum i_Direction)
{
  switch (i_Direction)
  {
    case Direction::Forward:
      return "Forward";
    case Direction::Backward:
      return "Backward";
    case Direction::Left:
      return "Left";
    case Direction::Right:
      return "Right";
    case Direction::Up:
      return "Up";
    case Direction::Down:
      return "Down";
    default:
      return "Unknown";
  }
}

// Puts Result constants into readable format
string Adapt(const Result::Enum i_result)
{
  switch (i_result)
  {
    case Result::ClientAlreadyConnected:
      return "ClientAlreadyConnected";
    case Result::ClientConnectionFailed:
      return "";
    case Result::CoLinearAxes:
      return "CoLinearAxes";
    case Result::InvalidDeviceName:
      return "InvalidDeviceName";
    case Result::InvalidDeviceOutputName:
      return "InvalidDeviceOutputName";
    case Result::InvalidHostName:
      return "InvalidHostName";
    case Result::InvalidIndex:
      return "InvalidIndex";
    case Result::InvalidLatencySampleName:
      return "InvalidLatencySampleName";
    case Result::InvalidMarkerName:
      return "InvalidMarkerName";
    case Result::InvalidMulticastIP:
      return "InvalidMulticastIP";
    case Result::InvalidSegmentName:
      return "InvalidSegmentName";
    case Result::InvalidSubjectName:
      return "InvalidSubjectName";
    case Result::LeftHandedAxes:
      return "LeftHandedAxes";
    case Result::NoFrame:
      return "NoFrame";
    case Result::NotConnected:
      return "NotConnected";
    case Result::NotImplemented:
      return "NotImplemented";
    case Result::ServerAlreadyTransmittingMulticast:
      return "ServerAlreadyTransmittingMulticast";
    case Result::ServerNotTransmittingMulticast:
      return "ServerNotTransmittingMulticast";
    case Result::Success:
      return "Success";
    case Result::Unknown:
      return "Unknown";
    default:
      return "unknown";
  }
}

// Class to hold a ROS publisher with status information
class SegmentPublisher
{
public:
  ros::Publisher pub;
  bool is_ready;
  tf::Transform calibration_pose;
  bool calibrated;
  SegmentPublisher() :
    is_ready(false), calibration_pose(tf::Pose::getIdentity()),
        calibrated(false)
  {
  }
  ;
};

typedef map<string, SegmentPublisher> SegmentMap;

class ViconReceiver
{
    
// Private variables
private:
  ros::NodeHandle nh;
  ros::NodeHandle nh_priv;
  // Diagnostic Updater
  diagnostic_updater::Updater diag_updater;
  double min_freq_;
  double max_freq_;
  diagnostic_updater::FrequencyStatus freq_status_;
  // Parameters:
  string stream_mode_;
  string host_name_;
  string tf_ref_frame_id_;
  string tracked_frame_suffix_;
  // Publisher
  ros::Publisher marker_pub_;
  ros::Publisher swarm_pub_; // Publisher for all swarmbots
  // TF Broadcaster
  tf::TransformBroadcaster tf_broadcaster_;
  //geometry_msgs::PoseStamped vicon_pose;
  tf::Transform flyer_transform;
  ros::Time now_time;
  // TODO: Make the following configurable:
  ros::ServiceServer m_grab_vicon_pose_service_server;
  ros::ServiceServer calibrate_segment_server_;
  //  ViconDataStreamSDK::CPP::Client MyClient;
  unsigned int lastFrameNumber;
  unsigned int frameCount;
  unsigned int droppedFrameCount;
  ros::Time time_datum;
  unsigned int frame_datum;
  unsigned int n_markers;
  unsigned int n_unlabeled_markers;
  bool segment_data_enabled;
  bool marker_data_enabled;
  bool unlabeled_marker_data_enabled;

  bool broadcast_tf_, publish_tf_, publish_markers_;

  bool grab_frames_;
  boost::thread grab_frames_thread_;
  SegmentMap segment_publishers_;
  boost::mutex segments_mutex_;
  std::vector<std::string> time_log_;

// Public methods
public:
  // Starts getting frames from datastream
  void startGrabbing()
  {
    grab_frames_ = true;
    // test grabbing in the main loop and run an asynchronous spinner instead
    grabThread();
    //grab_frames_thread_ = boost::thread(&ViconReceiver::grabThread, this);
  }

  void stopGrabbing()
  {
    grab_frames_ = false;
    //grab_frames_thread_.join();
  }

  // Constructor. Automatically sets up connection to vicon and starts publishing
  ViconReceiver() :
    nh_priv("~"), diag_updater(), min_freq_(0.1), max_freq_(1000),
        freq_status_(diagnostic_updater::FrequencyStatusParam(&min_freq_, &max_freq_)), stream_mode_("ClientPull"),
        host_name_(""), tf_ref_frame_id_("world"), tracked_frame_suffix_("vicon"),
        lastFrameNumber(0), frameCount(0), droppedFrameCount(0), frame_datum(0), n_markers(0), n_unlabeled_markers(0),
        marker_data_enabled(false), unlabeled_marker_data_enabled(false), grab_frames_(false)

  {
    // Diagnostics
    diag_updater.add("ViconReceiver Status", this, &ViconReceiver::diagnostics);
    diag_updater.add(freq_status_);
    diag_updater.setHardwareID("none");
    diag_updater.force_update();
    
    // Launchfile parameters
    nh_priv.param("stream_mode", stream_mode_, stream_mode_);
    nh_priv.param("datastream_hostport", host_name_, host_name_);
    nh_priv.param("tf_ref_frame_id", tf_ref_frame_id_, tf_ref_frame_id_);
    nh_priv.param("broadcast_transform", broadcast_tf_, true);
    nh_priv.param("publish_transform", publish_tf_, true);
    nh_priv.param("publish_markers", publish_markers_, true);
    
    // Initializes vicon
    if (init_vicon() == false){
      ROS_ERROR("Error while connecting to Vicon. Exiting now.");
      return;
    }
    
    // Service Server
    ROS_INFO("setting up grab_vicon_pose service server ... ");
    m_grab_vicon_pose_service_server = nh_priv.advertiseService("grab_vicon_pose", &ViconReceiver::grabPoseCallback,
                                                                this);

    // Calibration server
    ROS_INFO("setting up segment calibration service server ... ");
    calibrate_segment_server_ = nh_priv.advertiseService("calibrate_segment", &ViconReceiver::calibrateSegmentCallback,
                                                         this);

    // Publisher for swarmbots
    swarm_pub_ = nh.advertise<wvu_swarm_std_msgs::viconBotArray>("/viconArray", 10);
    
    // Publisher for individual points
    if(publish_markers_)
    {
      marker_pub_ = nh.advertise<vicon_bridge::Markers>(tracked_frame_suffix_ + "/markers", 10);
    }
    
    startGrabbing();
  }

  // Deconstructor
  ~ViconReceiver()
  {
    for (size_t i = 0; i < time_log_.size(); i++)
    {
      std::cout << time_log_[i] << std::endl;
    }
    if (shutdown_vicon() == false){
      ROS_ERROR("Error while shutting down Vicon.");
    }
  }

// Private methods
private:
  void diagnostics(diagnostic_updater::DiagnosticStatusWrapper& stat)
  {
    stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "OK");
    stat.add("latest VICON frame number", lastFrameNumber);
    stat.add("dropped frames", droppedFrameCount);
    stat.add("framecount", frameCount);
    stat.add("# markers", n_markers);
    stat.add("# unlabeled markers", n_unlabeled_markers);
  }

  // Function to initialize datastream 
  bool init_vicon()
  {
    ROS_INFO_STREAM("Connecting to Vicon DataStream SDK at " << host_name_ << " ...");

    ros::Duration d(1);
    Result::Enum result(Result::Unknown);

    // Attempts to find an MS API datastream until one is connected
    while (!msvcbridge::IsConnected().Connected)
    {
      msvcbridge::Connect(host_name_);
      ROS_INFO(".");
      d.sleep();
      ros::spinOnce();
      if (!ros::ok())
        return false;
    }
    
    // Connection was successful
    ROS_ASSERT(msvcbridge::IsConnected().Connected);
    ROS_INFO_STREAM("... connected!");

    // Sets the datastream to the mode specified in launchfile
    if (stream_mode_ == "ServerPush")
    {
      result = msvcbridge::SetStreamMode(StreamMode::ServerPush).Result;
    }
    else if (stream_mode_ == "ClientPull")
    {
      result = msvcbridge::SetStreamMode(StreamMode::ClientPull).Result;
    }
    else
    {
      ROS_FATAL("Unknown stream mode -- options are ServerPush, ClientPull");
      ros::shutdown();
    }

    ROS_INFO_STREAM("Setting Stream Mode to " << stream_mode_<< ": "<< Adapt(result));

    // Sets up axes with black magic
    msvcbridge::SetAxisMapping(Direction::Forward, Direction::Left, Direction::Up); // 'Z-up'
    Output_GetAxisMapping _Output_GetAxisMapping = msvcbridge::GetAxisMapping();

    // Prints results of axis mapping
    ROS_INFO_STREAM("Axis Mapping: X-" << Adapt(_Output_GetAxisMapping.XAxis) << " Y-"
        << Adapt(_Output_GetAxisMapping.YAxis) << " Z-" << Adapt(_Output_GetAxisMapping.ZAxis));

    // Presumably sets up the datastream to output data??
    msvcbridge::EnableSegmentData();
    ROS_ASSERT(msvcbridge::IsSegmentDataEnabled().Enabled);

    // Version logging
    Output_GetVersion _Output_GetVersion = msvcbridge::GetVersion();
    ROS_INFO_STREAM("Version: " << _Output_GetVersion.Major << "." << _Output_GetVersion.Minor << "."
        << _Output_GetVersion.Point);
    
    // At this point if still running, return true for success
    return true;
  }

  // Sets up a Boost thread for EACH TRACKED SUBJECT as its own publisher
  void createSegmentThread(const string subject_name, const string segment_name)
  {
    ROS_INFO("creating new object %s/%s ...",subject_name.c_str(), segment_name.c_str() );
    boost::mutex::scoped_lock lock(segments_mutex_);
    
    // Publish the segment addressed by "subject/segment"
    SegmentPublisher & spub = segment_publishers_[subject_name + "/" + segment_name];

    // we don't need the lock anymore, since rest is protected by is_ready
    lock.unlock();

    if(publish_tf_)
    {
      spub.pub = nh.advertise<geometry_msgs::TransformStamped>(
            tracked_frame_suffix_ + "/" + subject_name + "/" + segment_name, 10);
    }
    
    // Try to get zero pose from parameter server
    string param_suffix(subject_name + "/" + segment_name + "/zero_pose/");
    double qw, qx, qy, qz, x, y, z;
    bool have_params = true;
    have_params = have_params && nh_priv.getParam(param_suffix + "orientation/w", qw);
    have_params = have_params && nh_priv.getParam(param_suffix + "orientation/x", qx);
    have_params = have_params && nh_priv.getParam(param_suffix + "orientation/y", qy);
    have_params = have_params && nh_priv.getParam(param_suffix + "orientation/z", qz);
    have_params = have_params && nh_priv.getParam(param_suffix + "position/x", x);
    have_params = have_params && nh_priv.getParam(param_suffix + "position/y", y);
    have_params = have_params && nh_priv.getParam(param_suffix + "position/z", z);

    // If successful, load the pose. Otherwise load arbitrary calibration
    if (have_params)
    {
      ROS_INFO("loaded zero pose for %s/%s", subject_name.c_str(), segment_name.c_str());
      spub.calibration_pose.setRotation(tf::Quaternion(qx, qy, qz, qw));
      spub.calibration_pose.setOrigin(tf::Vector3(x, y, z));
      spub.calibration_pose = spub.calibration_pose.inverse();
    }
    else
    {
      ROS_WARN("unable to load zero pose for %s/%s", subject_name.c_str(), segment_name.c_str());
      spub.calibration_pose.setIdentity();
    }

    // I'm not completely sure how is_ready gets set to true, but it happens somehow
    spub.is_ready = true;
    ROS_INFO("... done, advertised as \" %s/%s/%s\" ", tracked_frame_suffix_.c_str(), subject_name.c_str(), segment_name.c_str());

  }

  // Wraps createSegmentThread as a Boost thread
  void createSegment(const string subject_name, const string segment_name)
  {
    boost::thread(&ViconReceiver::createSegmentThread, this, subject_name, segment_name);
  }

  // Tries to grab frames from the datastream
  void grabThread()
  {
    ros::Duration d(1.0 / 240.0);
//    ros::Time last_time = ros::Time::now();
//    double fps = 100.;
//    ros::Duration diff;
//    std::stringstream time_log;
    
    while (ros::ok() && grab_frames_)
    {
      // Tries to grab a frame from the datastream
      while (msvcbridge::GetFrame().Result != Result::Success && ros::ok())
      {
        ROS_INFO("getFrame returned false");
        d.sleep();
      }
      now_time = ros::Time::now();
//      diff = now_time-last_time;
//      fps = 1.0/(0.9/fps + 0.1*diff.toSec());
//      time_log.clear();
//      time_log.str("");
//      time_log <<"timings: dt="<<diff<<" fps=" <<fps;
//      time_log_.push_back(time_log.str());
//      last_time = now_time;

      bool was_new_frame = process_frame();
      ROS_WARN_COND(!was_new_frame, "grab frame returned false");

      diag_updater.update();
    }
  }

  // Shuts down the datastream api. Returns true on success.
  bool shutdown_vicon()
  {
    ROS_INFO_STREAM("stopping grabbing thread");
    stopGrabbing();
    ROS_INFO_STREAM("Disconnecting from Vicon DataStream SDK");
    msvcbridge::Disconnect();
    ROS_ASSERT(!msvcbridge::IsConnected().Connected);
    ROS_INFO_STREAM("... disconnected.");
    return true;
  }

  bool process_frame()
  {
    static ros::Time lastTime;
    Output_GetFrameNumber OutputFrameNum = msvcbridge::GetFrameNumber();

    //frameCount++;
    //ROS_INFO_STREAM("Grabbed a frame: " << OutputFrameNum.FrameNumber);
    int frameDiff = 0;
    if (lastFrameNumber != 0)
    {
      frameDiff = OutputFrameNum.FrameNumber - lastFrameNumber;
      frameCount += frameDiff;
      if ((frameDiff) > 1)
      {
        droppedFrameCount += frameDiff;
        double droppedFramePct = (double)droppedFrameCount / frameCount * 100;
        ROS_DEBUG_STREAM(frameDiff << " more (total " << droppedFrameCount << "/" << frameCount << ", "
            << droppedFramePct << "%) frame(s) dropped. Consider adjusting rates.");
      }
    }
    lastFrameNumber = OutputFrameNum.FrameNumber;

    if (frameDiff == 0)
    {
      return false;
    }
    else
    {
      freq_status_.tick();
      ros::Duration vicon_latency(msvcbridge::GetLatencyTotal().Total);

      // Publish each segment as its own topic
      if(publish_tf_ || broadcast_tf_)
      {
        process_subjects(now_time - vicon_latency);
      }

      // Publish individual Markers as one topic
      if(publish_markers_)
      {
        process_markers(now_time - vicon_latency, lastFrameNumber);
      }

      lastTime = now_time;
      return true;
    }
  }

  // Iterates through subjects in a frame, creates a Segment Publisher for each
  void process_subjects(const ros::Time& frame_time)
  {
    string tracked_frame, subject_name, segment_name;
    
    geometry_msgs::TransformStamped geoStampedTf;
    
    // No idea what this does, too afraid to delete it
    static unsigned int cnt = 0;
    
    // Set up variables to construct transforms and bots
    tf::Transform transform;
    wvu_swarm_std_msgs::viconBotArray botArray;

    // Find the number of subjects being tracked
    unsigned int n_subjects = msvcbridge::GetSubjectCount().SubjectCount;
    
    // Iterate through subjects
    for (unsigned int i_subjects = 0; i_subjects < n_subjects; i_subjects++)
    {
      // Finds this subject's name
      subject_name = msvcbridge::GetSubjectName(i_subjects).SubjectName;
      
      // Only continue if this is the right name for us
      // TODO: make this parameter editable by the launchfile
      if(subject_name.find("swarmbot_") != string::npos)
      {
        // Finds number of segments of this subject
        unsigned int n_segments = msvcbridge::GetSegmentCount(subject_name).SegmentCount;

        for (unsigned int i_segments = 0; i_segments < n_segments; i_segments++)
        {
          // Gets the name of this segment
          segment_name = msvcbridge::GetSegmentName(subject_name, i_segments).SegmentName;
          
          // Get the id of this segment
          //std::uint8_t botId[2];
          //botId[0] = (std::uint8_t)segment_name.at(segment_name.find("_")+1); // Probably better ways to do this, sorry
          //botId[1] = (std::uint8_t)segment_name.at(segment_name.find("_")+2);
          //std::vector<std::uint8_t> botId = {(std::uint8_t)segment_name.at(segment_name.find("_")+1),
          //                                   (std::uint8_t)segment_name.at(segment_name.find("_")+2)};
          boost::array<std::uint8_t, 2> botId;
          botId[0] = (std::uint8_t)segment_name.at(segment_name.find("_")+1); // Probably better ways to do this, sorry
          if(segment_name.length() > segment_name.find("_")+2)
            botId[1] = (std::uint8_t)segment_name.at(segment_name.find("_")+2);
          else
            botId[1] = (std::uint8_t)' ';

          // Grabs orientation and translation from the datastream
          Output_GetSegmentGlobalTranslation trans = msvcbridge::GetSegmentGlobalTranslation(subject_name, segment_name);
          Output_GetSegmentGlobalRotationQuaternion quat = msvcbridge::GetSegmentGlobalRotationQuaternion(subject_name,
                                                                                                          segment_name);

          if (trans.Result == Result::Success && quat.Result == Result::Success)
          {
            if (!trans.Occluded && !quat.Occluded)
            {
              // Define the transform using the datastream info
              transform.setOrigin(tf::Vector3(trans.Translation[0] / 1000, trans.Translation[1] / 1000,
                                                    trans.Translation[2] / 1000));
              transform.setRotation(tf::Quaternion(quat.Rotation[0], quat.Rotation[1], quat.Rotation[2],
                                                         quat.Rotation[3]));

              tracked_frame = tracked_frame_suffix_ + "/" + subject_name + "/" + segment_name;
              
              // Stamp the transform
              tf::StampedTransform stampTf(transform, frame_time, tf_ref_frame_id_, tracked_frame);
              
              // Convert to geometry_msgs type
              tf::transformStampedTFToMsg(stampTf, geoStampedTf);
              
              // Create a swarmbot, add to vector
              wvu_swarm_std_msgs::viconBot thisBot;
              thisBot.botId = botId;
              thisBot.botPose = geoStampedTf;
              botArray.poseVect.push_back(thisBot);
              
              // Original code to do one topic per segment
              /*
              boost::mutex::scoped_try_lock lock(segments_mutex_);

              if (lock.owns_lock())
              {
                pub_it = segment_publishers_.find(subject_name + "/" + segment_name);
                if (pub_it != segment_publishers_.end())
                {
                  SegmentPublisher & seg = pub_it->second;
                  //ros::Time thisTime = now_time - ros::Duration(latencyInMs / 1000);

                  if (seg.is_ready)
                  {
                    transform = transform * seg.calibration_pose;
                    transforms.push_back(tf::StampedTransform(transform, frame_time, tf_ref_frame_id_, tracked_frame));
  //                  transform = tf::StampedTransform(flyer_transform, frame_time, tf_ref_frame_id_, tracked_frame);
  //                  tf_broadcaster_.sendTransform(transform);

                    if(publish_tf_)
                    {
                      tf::transformStampedTFToMsg(transforms.back(), *pose_msg);
                      seg.pub.publish(pose_msg);
                    }
                  }
                }
                else
                {
                  lock.unlock();
                  createSegment(subject_name, segment_name);
                }
              }
              */
            }
            else
            {
              if (cnt % 100 == 0)
                ROS_WARN_STREAM("" << subject_name <<" occluded, not publishing... " );
            }
          }
          else
          {
            ROS_WARN("GetSegmentGlobalTranslation/Rotation failed (result = %s, %s), not publishing...",
                Adapt(trans.Result).c_str(), Adapt(quat.Result).c_str());
          }
        }
      }
    }
    
    /*
    if(broadcast_tf_)
    {
      tf_broadcaster_.sendTransform(transforms);
    }
    */
    
    // Broadcast??
    swarm_pub_.publish(botArray);
    
    cnt++;
  }

  void process_markers(const ros::Time& frame_time, unsigned int vicon_frame_num)
  {
    if (marker_pub_.getNumSubscribers() > 0)
    {
      if (not marker_data_enabled)
      {
        msvcbridge::EnableMarkerData();
        ROS_ASSERT(msvcbridge::IsMarkerDataEnabled().Enabled);
        marker_data_enabled = true;
      }
      if (not unlabeled_marker_data_enabled)
      {
        msvcbridge::EnableUnlabeledMarkerData();
        ROS_ASSERT(msvcbridge::IsUnlabeledMarkerDataEnabled().Enabled);
        unlabeled_marker_data_enabled = true;
      }
      n_markers = 0;
      vicon_bridge::Markers markers_msg;
      markers_msg.header.stamp = frame_time;
      markers_msg.frame_number = vicon_frame_num;
      // Count the number of subjects
      unsigned int SubjectCount = msvcbridge::GetSubjectCount().SubjectCount;
      // Get labeled markers
      for (unsigned int SubjectIndex = 0; SubjectIndex < SubjectCount; ++SubjectIndex)
      {
        std::string this_subject_name = msvcbridge::GetSubjectName(SubjectIndex).SubjectName;
        // Count the number of markers
        unsigned int num_subject_markers = msvcbridge::GetMarkerCount(this_subject_name).MarkerCount;
        n_markers += num_subject_markers;
        //std::cout << "    Markers (" << MarkerCount << "):" << std::endl;
        for (unsigned int MarkerIndex = 0; MarkerIndex < num_subject_markers; ++MarkerIndex)
        {
          vicon_bridge::Marker this_marker;
          this_marker.marker_name = msvcbridge::GetMarkerName(this_subject_name, MarkerIndex).MarkerName;
          this_marker.subject_name = this_subject_name;
          this_marker.segment_name
              = msvcbridge::GetMarkerParentName(this_subject_name, this_marker.marker_name).SegmentName;

          // Get the global marker translation
          Output_GetMarkerGlobalTranslation _Output_GetMarkerGlobalTranslation =
              msvcbridge::GetMarkerGlobalTranslation(this_subject_name, this_marker.marker_name);

          this_marker.translation.x = _Output_GetMarkerGlobalTranslation.Translation[0];
          this_marker.translation.y = _Output_GetMarkerGlobalTranslation.Translation[1];
          this_marker.translation.z = _Output_GetMarkerGlobalTranslation.Translation[2];
          this_marker.occluded = _Output_GetMarkerGlobalTranslation.Occluded;

          markers_msg.markers.push_back(this_marker);
        }
      }
      // get unlabeled markers
      unsigned int UnlabeledMarkerCount = msvcbridge::GetUnlabeledMarkerCount().MarkerCount;
      //ROS_INFO("# unlabeled markers: %d", UnlabeledMarkerCount);
      n_markers += UnlabeledMarkerCount;
      n_unlabeled_markers = UnlabeledMarkerCount;
      for (unsigned int UnlabeledMarkerIndex = 0; UnlabeledMarkerIndex < UnlabeledMarkerCount; ++UnlabeledMarkerIndex)
      {
        // Get the global marker translation
        Output_GetUnlabeledMarkerGlobalTranslation _Output_GetUnlabeledMarkerGlobalTranslation =
            msvcbridge::GetUnlabeledMarkerGlobalTranslation(UnlabeledMarkerIndex);

        if (_Output_GetUnlabeledMarkerGlobalTranslation.Result == Result::Success)
        {
          vicon_bridge::Marker this_marker;
          this_marker.translation.x = _Output_GetUnlabeledMarkerGlobalTranslation.Translation[0];
          this_marker.translation.y = _Output_GetUnlabeledMarkerGlobalTranslation.Translation[1];
          this_marker.translation.z = _Output_GetUnlabeledMarkerGlobalTranslation.Translation[2];
          this_marker.occluded = false; // unlabeled markers can't be occluded
          markers_msg.markers.push_back(this_marker);
        }
        else
        {
          ROS_WARN("GetUnlabeledMarkerGlobalTranslation failed (result = %s)",
              Adapt(_Output_GetUnlabeledMarkerGlobalTranslation.Result).c_str());

        }
      }
      marker_pub_.publish(markers_msg);
    }
  }

  bool grabPoseCallback(vicon_bridge::viconGrabPose::Request& req, vicon_bridge::viconGrabPose::Response& resp)
  {
    ROS_INFO("Got request for a VICON pose");
    tf::TransformListener tf_listener;
    tf::StampedTransform transform;
    tf::Quaternion orientation(0, 0, 0, 0);
    tf::Vector3 position(0, 0, 0);

    string tracked_segment = tracked_frame_suffix_ + "/" + req.subject_name + "/" + req.segment_name;

    // Gather data:
    int N = req.n_measurements;
    int n_success = 0;
    ros::Duration timeout(0.1);
    ros::Duration poll_period(1.0 / 240.0);

    for (int k = 0; k < N; k++)
    {
      try
      {
        if (tf_listener.waitForTransform(tf_ref_frame_id_, tracked_segment, ros::Time::now(), timeout, poll_period))
        {
          tf_listener.lookupTransform(tf_ref_frame_id_, tracked_segment, ros::Time(0), transform);
          orientation += transform.getRotation();
          position += transform.getOrigin();
          n_success++;
        }
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
        //    		resp.success = false;
        //    		return false; // TODO: should we really bail here, or just try again?
      }
    }

    // Average the data
    orientation /= n_success;
    orientation.normalize();
    position /= n_success;

    // copy what we used to service call response:
    resp.success = true;
    resp.pose.header.stamp = ros::Time::now();
    resp.pose.header.frame_id = tf_ref_frame_id_;
    resp.pose.pose.position.x = position.x();
    resp.pose.pose.position.y = position.y();
    resp.pose.pose.position.z = position.z();
    resp.pose.pose.orientation.w = orientation.w();
    resp.pose.pose.orientation.x = orientation.x();
    resp.pose.pose.orientation.y = orientation.y();
    resp.pose.pose.orientation.z = orientation.z();

    return true;
  }

  bool calibrateSegmentCallback(vicon_bridge::viconCalibrateSegment::Request& req,
                                vicon_bridge::viconCalibrateSegment::Response& resp)
  {

    std::string full_name = req.subject_name + "/" + req.segment_name;
    ROS_INFO("trying to calibrate %s", full_name.c_str());

    SegmentMap::iterator seg_it = segment_publishers_.find(full_name);

    if (seg_it == segment_publishers_.end())
    {
      ROS_WARN("frame %s not found --> not calibrating", full_name.c_str());
      resp.success = false;
      resp.status = "segment " + full_name + " not found";
      return false;
    }

    SegmentPublisher & seg = seg_it->second;

    if (seg.calibrated)
    {
      ROS_INFO("%s already calibrated, deleting old calibration", full_name.c_str());
      seg.calibration_pose.setIdentity();
    }

    vicon_bridge::viconGrabPose::Request grab_req;
    vicon_bridge::viconGrabPose::Response grab_resp;

    grab_req.n_measurements = req.n_measurements;
    grab_req.subject_name = req.subject_name;
    grab_req.segment_name = req.segment_name;

    bool ret = grabPoseCallback(grab_req, grab_resp);

    if (!ret)
    {
      resp.success = false;
      resp.status = "error while grabbing pose from Vicon";
      return false;
    }

    tf::Transform t;
    t.setOrigin(tf::Vector3(grab_resp.pose.pose.position.x, grab_resp.pose.pose.position.y,
                            grab_resp.pose.pose.position.z - req.z_offset));
    t.setRotation(tf::Quaternion(grab_resp.pose.pose.orientation.x, grab_resp.pose.pose.orientation.y,
                                 grab_resp.pose.pose.orientation.z, grab_resp.pose.pose.orientation.w));

    seg.calibration_pose = t.inverse();

    // write zero_pose to parameter server
    string param_suffix(full_name + "/zero_pose/");
    nh_priv.setParam(param_suffix + "orientation/w", t.getRotation().w());
    nh_priv.setParam(param_suffix + "orientation/x", t.getRotation().x());
    nh_priv.setParam(param_suffix + "orientation/y", t.getRotation().y());
    nh_priv.setParam(param_suffix + "orientation/z", t.getRotation().z());

    nh_priv.setParam(param_suffix + "position/x", t.getOrigin().x());
    nh_priv.setParam(param_suffix + "position/y", t.getOrigin().y());
    nh_priv.setParam(param_suffix + "position/z", t.getOrigin().z());

    ROS_INFO_STREAM("calibration completed");
    resp.pose = grab_resp.pose;
    resp.success = true;
    resp.status = "calibration successful";
    seg.calibrated = true;

    return true;
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vicon");
//  ViconReceiver vr;
//  ros::spin();

  ros::AsyncSpinner aspin(1);
  aspin.start();
  ViconReceiver vr;
  aspin.stop();
  return 0;
}
