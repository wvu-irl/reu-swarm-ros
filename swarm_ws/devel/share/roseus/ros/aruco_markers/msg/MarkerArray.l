;; Auto-generated. Do not edit!


(when (boundp 'aruco_markers::MarkerArray)
  (if (not (find-package "ARUCO_MARKERS"))
    (make-package "ARUCO_MARKERS"))
  (shadow 'MarkerArray (find-package "ARUCO_MARKERS")))
(unless (find-package "ARUCO_MARKERS::MARKERARRAY")
  (make-package "ARUCO_MARKERS::MARKERARRAY"))

(in-package "ROS")
;;//! \htmlinclude MarkerArray.msg.html


(defclass aruco_markers::MarkerArray
  :super ros::object
  :slots (_markers ))

(defmethod aruco_markers::MarkerArray
  (:init
   (&key
    ((:markers __markers) (let (r) (dotimes (i 0) (push (instance aruco_markers::Marker :init) r)) r))
    )
   (send-super :init)
   (setq _markers __markers)
   self)
  (:markers
   (&rest __markers)
   (if (keywordp (car __markers))
       (send* _markers __markers)
     (progn
       (if __markers (setq _markers (car __markers)))
       _markers)))
  (:serialization-length
   ()
   (+
    ;; aruco_markers/Marker[] _markers
    (apply #'+ (send-all _markers :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; aruco_markers/Marker[] _markers
     (write-long (length _markers) s)
     (dolist (elem _markers)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; aruco_markers/Marker[] _markers
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _markers (let (r) (dotimes (i n) (push (instance aruco_markers::Marker :init) r)) r))
     (dolist (elem- _markers)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get aruco_markers::MarkerArray :md5sum-) "c83561ca1ae4ac98039651009b8168f1")
(setf (get aruco_markers::MarkerArray :datatype-) "aruco_markers/MarkerArray")
(setf (get aruco_markers::MarkerArray :definition-)
      "aruco_markers/Marker[] markers

================================================================================
MSG: aruco_markers/Marker
std_msgs/Header header

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



(provide :aruco_markers/MarkerArray "c83561ca1ae4ac98039651009b8168f1")


