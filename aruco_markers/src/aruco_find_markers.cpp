/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2018, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
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
*   * Neither the name of the Willow Garage nor the names of its
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
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <geometry_msgs/Vector3.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// aruco api: https://docs.opencv.org/3.4/d9/d53/aruco_8hpp.html
#include <opencv2/aruco.hpp>
#include <aruco_markers/Marker.h>
#include <aruco_markers/MarkerArray.h>

#include <ros/console.h>
#include <iostream>
#include <geometry_msgs/Pose2D.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>
#define _USE_MATH_DEFINES // for math constants (M_PI)

cv::Mat image_; // drawn on image

void image_clk(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from <%s> to <bgr8>", msg->encoding.c_str());
  }

  image_ = cv_ptr->image;
  // cv::imshow("view", cv_ptr->image);
  // cv::waitKey(1);
}

int main(int _argc, char** _argv)
{
  ros::init(_argc, _argv, "find_marker_node");
  ros::NodeHandle nh;

  // retrieving parameters from launch file
  std::vector<double> cam_matrix_data;
  nh.getParam("/find_marker_node/camera_matrix/data", cam_matrix_data);

  std::vector<double> dist_coeffs_data;
  nh.getParam("/find_marker_node/distortion_coefficients/data", dist_coeffs_data);

  bool use_video_device;
  nh.getParam("/find_marker_node/use_video_device", use_video_device);

  std::string image_topic;
  nh.getParam("/find_marker_node/image_topic", image_topic);

  int video_device_num; // video device number
  nh.getParam("/find_marker_node/video_device_num", video_device_num);

  int lr; // loop rate
  nh.getParam("/find_marker_node/loop_rate", lr);
  
  int camWide;
  int camHigh; // resolution width and height
  nh.getParam("/find_marker_node/resolution_width", camWide);
  nh.getParam("/find_marker_node/resolution_height", camHigh);
  
  bool autoFocus;
  nh.getParam("/find_marker_node/auto_focus", autoFocus);

  float marker_size;
  nh.getParam("/find_marker_node/marker_size", marker_size);

  image_transport::ImageTransport it(nh);

  // Publishers
  aruco_markers::MarkerArray marker_array;
  ros::Publisher pub_marker = nh.advertise<aruco_markers::MarkerArray>("/markers", 1);

  // raw/original image publisher
  sensor_msgs::ImagePtr img_raw_msg;
  image_transport::Publisher pub_raw_img = it.advertise("/camera/image_raw", camHigh*camWide);

  // drawn marker on image publisher
  sensor_msgs::ImagePtr img_marker_msg;
  image_transport::Publisher pub_marker_img = it.advertise("/camera/image_marker", camHigh*camWide);

  image_transport::Subscriber sub_image;
  if (!use_video_device)
  {
    // subscribe to camera/image topic
    sub_image = it.subscribe(image_topic, camWide*camHigh, image_clk, ros::VoidPtr());
  }

  // predefined aruco marker dictionary
  cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

  // aruco markers parameters
  cv::Ptr<cv::aruco::DetectorParameters> parameters = new cv::aruco::DetectorParameters();
  std::vector<int> marker_ids; // ids of markers detected
  std::vector<std::vector<cv::Point2f>> marker_corners; // vector locations of detected markers' corners (clockwise order; pixel coordinates)
  std::vector<cv::Vec3d> rvecs, tvecs; // rotation & translation vectors of detected markers

  // calibrated camera parameters
  cv::Mat camera_matrix(cv::Size(3,3), CV_64F, cam_matrix_data.data());
  cv::Mat distort_coeffs(cv::Size(1,5), CV_64F, dist_coeffs_data.data());

  cv::VideoCapture capture;
  if (use_video_device)
  {
    // opens up camera device
    capture.open(video_device_num);
  }

  //use capture parameters from launchfile
  capture.set(cv::CAP_PROP_FRAME_WIDTH, camWide);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, camHigh);
  capture.set(cv::CAP_PROP_AUTOFOCUS, autoFocus);	
  
  //change parameters to better handle full resolution
  parameters->minMarkerPerimeterRate = 0.007; //smallest id will be 1920*0.007=13px
  parameters->maxMarkerPerimeterRate = 0.1; //largest id will be 1920*0.1=192px
  parameters->errorCorrectionRate = 0.9; //higher alpha errors, lower beta errors
  parameters->adaptiveThreshConstant = 7; //default
  parameters->maxErroneousBitsInBorderRate = 0.35; //default

  cv::Mat raw_img; // copied raw imag

  uint count = 0;
  ros::Rate loop_rate(lr);

  while (nh.ok())
  {
    if (use_video_device)
    {
      // capture new image frame
      capture >> image_;
    }

    if(!image_.empty() && cv::sum(image_-raw_img)[0] != 0) // checks if a new image
    {
      image_.copyTo(raw_img);

      marker_array.markers.clear();
      marker_ids.clear();

      cv::aruco::detectMarkers(image_, dict, marker_corners, marker_ids, parameters=parameters);

      if (marker_ids.size() > 0)
      {
        aruco_markers::Marker marker;
        geometry_msgs::Pose2D pixel_corners;

        cv::aruco::drawDetectedMarkers(image_, marker_corners, marker_ids);
        cv::aruco::estimatePoseSingleMarkers(marker_corners, marker_size, camera_matrix, distort_coeffs, rvecs, tvecs);

        // publishes to /markers topic
        for(int i = 0; i < marker_ids.size(); ++i)
        {
          marker.header.frame_id = "aruco_markers";
          marker.header.stamp = ros::Time::now();
          marker.header.seq = count;

          marker.id = marker_ids[i];

          marker.rvec.x = rvecs[i][0];
          marker.rvec.y = rvecs[i][1];
          marker.rvec.z = rvecs[i][2];

          double angle = sqrt(marker.rvec.x*marker.rvec.x
                              + marker.rvec.y*marker.rvec.y
                              + marker.rvec.z*marker.rvec.z);
          double x = marker.rvec.x/angle;
          double y = marker.rvec.y/angle;
          double z = marker.rvec.z/angle;

          // from camera to marker ref frame
          tf2::Quaternion q(x*sin(angle/2),
                                       y*sin(angle/2),
                                       z*sin(angle/2),
                                       cos(angle/2));

          double roll, pitch, yaw;
           // roll (x-axis rotation)
         	double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
         	double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
         	roll = atan2(sinr_cosp, cosr_cosp);

         	// pitch (y-axis rotation)
         	double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
         	if (fabs(sinp) >= 1)
         		pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
         	else
         		pitch = asin(sinp);

         	// yaw (z-axis rotation)
         	double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
         	double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
         	yaw = atan2(siny_cosp, cosy_cosp);

          marker.rpy.x = roll;
          marker.rpy.y = pitch;
          marker.rpy.z = yaw;

          marker.tvec.x = tvecs[i][0];
          marker.tvec.y = tvecs[i][1];
          marker.tvec.z = tvecs[i][2];

          pixel_corners.x = marker_corners[i][0].x;
          pixel_corners.y = marker_corners[i][0].y;
          marker.pixel_corners.push_back(pixel_corners);

          pixel_corners.x = marker_corners[i][1].x;
          pixel_corners.y = marker_corners[i][1].y;
          marker.pixel_corners.push_back(pixel_corners);

          pixel_corners.x = marker_corners[i][2].x;
          pixel_corners.y = marker_corners[i][2].y;
          marker.pixel_corners.push_back(pixel_corners);

          pixel_corners.x = marker_corners[i][3].x;
          pixel_corners.y = marker_corners[i][3].y;
          marker.pixel_corners.push_back(pixel_corners);

          marker_array.markers.push_back(marker);

          cv::aruco::drawAxis(image_, camera_matrix, distort_coeffs, rvecs[i], tvecs[i], marker_size);
        }
      }

      // cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
      // cv::imshow("image", image);
      // cv::waitKey(1);

      img_raw_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_img).toImageMsg();
      pub_raw_img.publish(img_raw_msg);

      img_marker_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_).toImageMsg();
      pub_marker_img.publish(img_marker_msg);
      if (marker_ids.size()) pub_marker.publish(marker_array);
      ++count;
	  }

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
