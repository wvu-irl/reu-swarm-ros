# Aruco Markers

## (1) Dependencies

* Core OpenCV 3 & Extra modules
  * You can follow the instructions found [here](https://github.com/wvu-irl/guides-and-resources/wiki/Core-OpenCV-and-Extra-Modules)

## (2) Marker Generation

This is a simple C++ program (not a ROS node).

* To compile, `cd aruco_markers/src/` & run the following in the terminal:

  ```
  g++ aruco_marker_generation.cpp -o aruco_marker_generation `pkg-config --libs opencv` -std=c++11
  ```

* To run:

  ```
  ./aruco_marker_generation
  ```

* When the OpenCV image window containing a marker image appears, hit the **_Enter_** key to continue to build the next one.

* The images will be saved inside the *aruco_markers/default_markers* directory.

## (3) Find Markers

* Follow the instructions [here](https://github.com/wvu-irl/guides-and-resources/wiki/Camera-Calibration) to retrieve the camera calibration _.yaml_ file.

  * **Note**: You need to get the camera calibration parameters to get an accurate marker location estimate.


* Put the *camera_calibration.yaml* file inside the *aruco_markers/launch/launch_params/* directory.

* Edit the *aruco_markers/launch/find_markers.launch* file

  * Edit the default parameters to your needs:

    ```
    <!-- Set to true if using image from video device
         Set to false if using image from ROS image topic -->
    <arg name="use_video_device" default="true" />
    <!-- If subscribing to ROS image topic, set above arg use_video_device to false -->
    <arg name="image_topic" default="/camera/color/image_raw" />

    <!-- video device number from /dev/videoX (e.g. /dev/video0) -->
    <arg name="video_device_num" default="0" />
    <!-- loop rate ~ frames per second -->
    <arg name="loop_rate" default="60" />
    <!-- size of markers in meters -->
    <arg name="marker_size" default="0.031" />
    ```

  * Update `line 11` to the correct filename (i.e. `<rosparam command="load" file="$(find aruco_markers)/launch/launch_params/usb_cam_calibration.yaml" />`)


* Compile with `catkin_make`.

* Run with `roslaunch aruco_markers find_markers.launch`

  * You will see the detected markers under the */markers* topic.

  * The raw image will be under the */camera/image_raw* topic and the drawn marker image under the */camera/image_marker* topic.
