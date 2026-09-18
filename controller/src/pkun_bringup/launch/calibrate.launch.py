"""Servo calibration bringup: driver only, no motion, no gait.

Nothing publishes joint commands here, so the robot holds whatever you send by
hand. That is exactly what you want while fitting horns.

  ros2 launch pkun_bringup calibrate.launch.py

Then, in another terminal:

  # release every servo so you can move the joints by hand
  ros2 service call /pkun/servo_driver/set_torque pkun_msgs/srv/SetTorque \\
      "{enable: false, name: []}"

  # re-enable and command one joint to zero
  ros2 service call /pkun/servo_driver/set_torque pkun_msgs/srv/SetTorque \\
      "{enable: true, name: []}"
  ros2 topic pub --once /pkun/joint_command pkun_msgs/msg/JointCommand \\
      "{name: ['fl_knee'], position: [0.0], duration: 1.0}"

  # nudge the zero until the link is truly straight, then read it back
  ros2 service call /pkun/servo_driver/set_trim pkun_msgs/srv/SetTrim \\
      "{name: 'fl_knee', trim: 0.035, persist: true}"
  ros2 param get /pkun/servo_driver servos.fl_knee.trim_deg

Copy the resulting trim_deg values into pkun_servo_driver/config/servos.yaml.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState

import lifecycle_msgs.msg


def generate_launch_description():
    servo_share = get_package_share_directory('pkun_servo_driver')
    namespace = LaunchConfiguration('namespace')

    servo_driver = LifecycleNode(
        package='pkun_servo_driver',
        executable='servo_driver_node',
        name='servo_driver',
        namespace=namespace,
        parameters=[os.path.join(servo_share, 'config', 'servos.yaml')],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='pkun'),
        servo_driver,
        EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(servo_driver),
                transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
            )
        ),
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=servo_driver,
                goal_state='inactive',
                entities=[
                    LogInfo(msg='servo driver ready for calibration'),
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(servo_driver),
                            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                        )
                    ),
                ],
            )
        ),
    ])
