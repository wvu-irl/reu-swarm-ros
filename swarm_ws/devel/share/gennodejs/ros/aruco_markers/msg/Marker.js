// Auto-generated. Do not edit!

// (in-package aruco_markers.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Marker {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.id = null;
      this.rvec = null;
      this.tvec = null;
      this.rpy = null;
      this.pixel_corners = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('id')) {
        this.id = initObj.id
      }
      else {
        this.id = 0;
      }
      if (initObj.hasOwnProperty('rvec')) {
        this.rvec = initObj.rvec
      }
      else {
        this.rvec = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('tvec')) {
        this.tvec = initObj.tvec
      }
      else {
        this.tvec = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('rpy')) {
        this.rpy = initObj.rpy
      }
      else {
        this.rpy = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('pixel_corners')) {
        this.pixel_corners = initObj.pixel_corners
      }
      else {
        this.pixel_corners = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Marker
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [id]
    bufferOffset = _serializer.int16(obj.id, buffer, bufferOffset);
    // Serialize message field [rvec]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rvec, buffer, bufferOffset);
    // Serialize message field [tvec]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.tvec, buffer, bufferOffset);
    // Serialize message field [rpy]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rpy, buffer, bufferOffset);
    // Serialize message field [pixel_corners]
    // Serialize the length for message field [pixel_corners]
    bufferOffset = _serializer.uint32(obj.pixel_corners.length, buffer, bufferOffset);
    obj.pixel_corners.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Pose2D.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Marker
    let len;
    let data = new Marker(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [id]
    data.id = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [rvec]
    data.rvec = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [tvec]
    data.tvec = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [rpy]
    data.rpy = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [pixel_corners]
    // Deserialize array length for message field [pixel_corners]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.pixel_corners = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.pixel_corners[i] = geometry_msgs.msg.Pose2D.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 24 * object.pixel_corners.length;
    return length + 78;
  }

  static datatype() {
    // Returns string type for a message object
    return 'aruco_markers/Marker';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '92623b17f39559a329eab104a28d98d2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Marker(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.id !== undefined) {
      resolved.id = msg.id;
    }
    else {
      resolved.id = 0
    }

    if (msg.rvec !== undefined) {
      resolved.rvec = geometry_msgs.msg.Vector3.Resolve(msg.rvec)
    }
    else {
      resolved.rvec = new geometry_msgs.msg.Vector3()
    }

    if (msg.tvec !== undefined) {
      resolved.tvec = geometry_msgs.msg.Vector3.Resolve(msg.tvec)
    }
    else {
      resolved.tvec = new geometry_msgs.msg.Vector3()
    }

    if (msg.rpy !== undefined) {
      resolved.rpy = geometry_msgs.msg.Vector3.Resolve(msg.rpy)
    }
    else {
      resolved.rpy = new geometry_msgs.msg.Vector3()
    }

    if (msg.pixel_corners !== undefined) {
      resolved.pixel_corners = new Array(msg.pixel_corners.length);
      for (let i = 0; i < resolved.pixel_corners.length; ++i) {
        resolved.pixel_corners[i] = geometry_msgs.msg.Pose2D.Resolve(msg.pixel_corners[i]);
      }
    }
    else {
      resolved.pixel_corners = []
    }

    return resolved;
    }
};

module.exports = Marker;
