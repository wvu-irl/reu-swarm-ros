;; Auto-generated. Do not edit!


(when (boundp 'aruco_markers::Marker)
  (if (not (find-package "ARUCO_MARKERS"))
    (make-package "ARUCO_MARKERS"))
  (shadow 'Marker (find-package "ARUCO_MARKERS")))
(unless (find-package "ARUCO_MARKERS::MARKER")
  (make-package "ARUCO_MARKERS::MARKER"))

(in-package "ROS")
;;//! \htmlinclude Marker.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass aruco_markers::Marker
  :super ros::object
  :slots (_header _id _rvec _tvec _rpy _pixel_corners ))

(defmethod aruco_markers::Marker
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:id __id) 0)
    ((:rvec __rvec) (instance geometry_msgs::Vector3 :init))
    ((:tvec __tvec) (instance geometry_msgs::Vector3 :init))
    ((:rpy __rpy) (instance geometry_msgs::Vector3 :init))
    ((:pixel_corners __pixel_corners) (let (r) (dotimes (i 0) (push (instance geometry_msgs::Pose2D :init) r)) r))
    )
   (send-super :init)
   (setq _header __header)
   (setq _id (round __id))
   (setq _rvec __rvec)
   (setq _tvec __tvec)
   (setq _rpy __rpy)
   (setq _pixel_corners __pixel_corners)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:id
   (&optional __id)
   (if __id (setq _id __id)) _id)
  (:rvec
   (&rest __rvec)
   (if (keywordp (car __rvec))
       (send* _rvec __rvec)
     (progn
       (if __rvec (setq _rvec (car __rvec)))
       _rvec)))
  (:tvec
   (&rest __tvec)
   (if (keywordp (car __tvec))
       (send* _tvec __tvec)
     (progn
       (if __tvec (setq _tvec (car __tvec)))
       _tvec)))
  (:rpy
   (&rest __rpy)
   (if (keywordp (car __rpy))
       (send* _rpy __rpy)
     (progn
       (if __rpy (setq _rpy (car __rpy)))
       _rpy)))
  (:pixel_corners
   (&rest __pixel_corners)
   (if (keywordp (car __pixel_corners))
       (send* _pixel_corners __pixel_corners)
     (progn
       (if __pixel_corners (setq _pixel_corners (car __pixel_corners)))
       _pixel_corners)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int16 _id
    2
    ;; geometry_msgs/Vector3 _rvec
    (send _rvec :serialization-length)
    ;; geometry_msgs/Vector3 _tvec
    (send _tvec :serialization-length)
    ;; geometry_msgs/Vector3 _rpy
    (send _rpy :serialization-length)
    ;; geometry_msgs/Pose2D[] _pixel_corners
    (apply #'+ (send-all _pixel_corners :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int16 _id
       (write-word _id s)
     ;; geometry_msgs/Vector3 _rvec
       (send _rvec :serialize s)
     ;; geometry_msgs/Vector3 _tvec
       (send _tvec :serialize s)
     ;; geometry_msgs/Vector3 _rpy
       (send _rpy :serialize s)
     ;; geometry_msgs/Pose2D[] _pixel_corners
     (write-long (length _pixel_corners) s)
     (dolist (elem _pixel_corners)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int16 _id
     (setq _id (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; geometry_msgs/Vector3 _rvec
     (send _rvec :deserialize buf ptr-) (incf ptr- (send _rvec :serialization-length))
   ;; geometry_msgs/Vector3 _tvec
     (send _tvec :deserialize buf ptr-) (incf ptr- (send _tvec :serialization-length))
   ;; geometry_msgs/Vector3 _rpy
     (send _rpy :deserialize buf ptr-) (incf ptr- (send _rpy :serialization-length))
   ;; geometry_msgs/Pose2D[] _pixel_corners
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _pixel_corners (let (r) (dotimes (i n) (push (instance geometry_msgs::Pose2D :init) r)) r))
     (dolist (elem- _pixel_corners)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get aruco_markers::Marker :md5sum-) "92623b17f39559a329eab104a28d98d2")
(setf (get aruco_markers::Marker :datatype-) "aruco_markers/Marker")
(setf (get aruco_markers::Marker :definition-)
      "std_msgs/Header header

# marker id
int16 id

# 3D space
geometry_msgs/Vector3 rvec
geometry_msgs/Vector3 tvec

# roll, pitch, yaw
geometry_msgs/Vector3 rpy

# pixel coordinates of corners (theta from Pose2D not used)
geometry_msgs/Pose2D[] pixel_corners

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Pose2D
# This expresses a position and orientation on a 2D manifold.

float64 x
float64 y
float64 theta
")



(provide :aruco_markers/Marker "92623b17f39559a329eab104a28d98d2")


