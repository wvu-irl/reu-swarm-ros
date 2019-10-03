#include <ros/ros.h>
#include <math.h>

#include <std_msgs/Float64.h>
#include <sensor_msgs/JointState.h>
#include <wvu_swarm_std_msgs/robot_command.h>

#define MAX_SPEED 1
#define EFFORT_MULT 0.005

static double left_speed, right_speed;
static double left_speed_setpoint, right_speed_setpoint;

void update_joints(sensor_msgs::JointState js)
{
  size_t joint_left, joint_right;
  if (js.name[0].compare("base_link_to_left_wheel") == 0)
  {
    joint_left = 0;
    joint_right = 1;
  }
  else
  {
    joint_left = 1;
    joint_right = 0;
  }

  left_speed = js.velocity[joint_left];
  right_speed = js.velocity[joint_right];
}

void update_exe(wvu_swarm_std_msgs::robot_command rc)
{
  double theta = -rc.theta;
  double vel = rc.r;

  left_speed_setpoint = vel * sin(theta);
  right_speed_setpoint = vel * sin(rc.theta);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "robot_controller");
  ros::NodeHandle n;

  left_speed = 0.5;
  right_speed = 0.5;

  ros::Subscriber joint_states = n.subscribe("joint_states", 1000, update_joints);
  ros::Subscriber exe = n.subscribe("final_execute", 1000, update_exe);

  ros::Publisher left_out = n.advertise<std_msgs::Float64>("left_wheel_velocity_controller/command", 1000);
  ros::Publisher right_out = n.advertise<std_msgs::Float64>("right_wheel_velocity_controller/command", 1000);

  while (ros::ok())
  {
    ros::spinOnce();

    double eff_left = (left_speed_setpoint - left_speed) * EFFORT_MULT;
    double eff_right = (right_speed_setpoint - right_speed) * EFFORT_MULT;

    std_msgs::Float64 left_flt, right_flt;
    left_flt.data = eff_left;
    right_flt.data = eff_right;

    left_out.publish(left_flt);
    right_out.publish(right_flt);
  }

  return 0;
}
