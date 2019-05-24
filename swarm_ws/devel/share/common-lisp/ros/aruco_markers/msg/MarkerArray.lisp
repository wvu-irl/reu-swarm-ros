; Auto-generated. Do not edit!


(cl:in-package aruco_markers-msg)


;//! \htmlinclude MarkerArray.msg.html

(cl:defclass <MarkerArray> (roslisp-msg-protocol:ros-message)
  ((markers
    :reader markers
    :initarg :markers
    :type (cl:vector aruco_markers-msg:Marker)
   :initform (cl:make-array 0 :element-type 'aruco_markers-msg:Marker :initial-element (cl:make-instance 'aruco_markers-msg:Marker))))
)

(cl:defclass MarkerArray (<MarkerArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MarkerArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MarkerArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name aruco_markers-msg:<MarkerArray> is deprecated: use aruco_markers-msg:MarkerArray instead.")))

(cl:ensure-generic-function 'markers-val :lambda-list '(m))
(cl:defmethod markers-val ((m <MarkerArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader aruco_markers-msg:markers-val is deprecated.  Use aruco_markers-msg:markers instead.")
  (markers m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MarkerArray>) ostream)
  "Serializes a message object of type '<MarkerArray>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'markers))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'markers))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MarkerArray>) istream)
  "Deserializes a message object of type '<MarkerArray>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'markers) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'markers)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'aruco_markers-msg:Marker))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MarkerArray>)))
  "Returns string type for a message object of type '<MarkerArray>"
  "aruco_markers/MarkerArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MarkerArray)))
  "Returns string type for a message object of type 'MarkerArray"
  "aruco_markers/MarkerArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MarkerArray>)))
  "Returns md5sum for a message object of type '<MarkerArray>"
  "c83561ca1ae4ac98039651009b8168f1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MarkerArray)))
  "Returns md5sum for a message object of type 'MarkerArray"
  "c83561ca1ae4ac98039651009b8168f1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MarkerArray>)))
  "Returns full string definition for message of type '<MarkerArray>"
  (cl:format cl:nil "aruco_markers/Marker[] markers~%~%================================================================================~%MSG: aruco_markers/Marker~%std_msgs/Header header~%~%# marker id~%int16 id~%~%# 3D space~%geometry_msgs/Vector3 rvec~%geometry_msgs/Vector3 tvec~%~%# roll, pitch, yaw~%geometry_msgs/Vector3 rpy~%~%# pixel coordinates of corners (theta from Pose2D not used)~%geometry_msgs/Pose2D[] pixel_corners~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Pose2D~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MarkerArray)))
  "Returns full string definition for message of type 'MarkerArray"
  (cl:format cl:nil "aruco_markers/Marker[] markers~%~%================================================================================~%MSG: aruco_markers/Marker~%std_msgs/Header header~%~%# marker id~%int16 id~%~%# 3D space~%geometry_msgs/Vector3 rvec~%geometry_msgs/Vector3 tvec~%~%# roll, pitch, yaw~%geometry_msgs/Vector3 rpy~%~%# pixel coordinates of corners (theta from Pose2D not used)~%geometry_msgs/Pose2D[] pixel_corners~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Pose2D~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MarkerArray>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'markers) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MarkerArray>))
  "Converts a ROS message object to a list"
  (cl:list 'MarkerArray
    (cl:cons ':markers (markers msg))
))
