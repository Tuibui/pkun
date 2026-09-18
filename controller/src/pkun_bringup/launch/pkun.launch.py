"""Bring up the whole P-kun stack.

Replaces the "initial connection" node idea: connecting to hardware is a
lifecycle transition owned by the driver, and this file is what drives that
transition in the right order.

Order matters. The servo driver is configured (which opens I2C and probes both
PCA9685 boards) and only activated once that succeeds. Motion, teleop and
expression are plain nodes -- if they start early they just publish into a
topic nobody is reading yet, which is harmless. The reverse is not: activating
servos before the motion node has a pose to send would leave the legs holding
whatever the horns powered up at.

  ros2 launch pkun_bringup pkun.launch.py
  ros2 launch pkun_bringup pkun.launch.py display_backend:=ssd1306
  ros2 launch pkun_bringup pkun.launch.py use_joy:=false
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState

import lifecycle_msgs.msg


def generate_launch_description():
    servo_share = get_package_share_directory('pkun_servo_driver')
    motion_share = get_package_share_directory('pkun_motion')
    teleop_share = get_package_share_directory('pkun_teleop')
    expression_share = get_package_share_directory('pkun_expression')

    namespace = LaunchConfiguration('namespace')
    use_joy = LaunchConfiguration('use_joy')
    display_backend = LaunchConfiguration('display_backend')
    joy_device = LaunchConfiguration('joy_device')

    args = [
        DeclareLaunchArgument(
            'namespace', default_value='pkun',
            description='Namespace for every P-kun node.'),
        DeclareLaunchArgument(
            'use_joy', default_value='true',
            description='Start joy_linux. Turn off to drive from the keyboard '
                        'or a bag instead.'),
        DeclareLaunchArgument(
            'display_backend', default_value='console',
            description="Face display: 'console' or 'ssd1306'."),
        DeclareLaunchArgument(
            'joy_device', default_value='',
            description='Joystick device path. Empty = first pad found.'),
    ]

    servo_driver = LifecycleNode(
        package='pkun_servo_driver',
        executable='servo_driver_node',
        name='servo_driver',
        namespace=namespace,
        parameters=[os.path.join(servo_share, 'config', 'servos.yaml')],
        output='screen',
    )

    # Unconfigured -> inactive. This is where the I2C bus is opened and both
    # boards are probed; a wiring fault fails here, loudly, before anything moves.
    configure_servos = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(servo_driver),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    # Inactive -> active, but only after configure actually reported success.
    activate_servos = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=servo_driver,
            goal_state='inactive',
            entities=[
                LogInfo(msg='servo driver configured, activating'),
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(servo_driver),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    motion = Node(
        package='pkun_motion',
        executable='motion_node',
        name='motion',
        namespace=namespace,
        parameters=[os.path.join(motion_share, 'config', 'motion.yaml')],
        output='screen',
    )

    expression = Node(
        package='pkun_expression',
        executable='expression_node',
        name='expression',
        namespace=namespace,
        parameters=[
            os.path.join(expression_share, 'config', 'expression.yaml'),
            {'display.backend': display_backend},
        ],
        output='screen',
    )

    teleop = Node(
        package='pkun_teleop',
        executable='joy_teleop_node',
        name='joy_teleop',
        namespace=namespace,
        parameters=[os.path.join(teleop_share, 'config', 'joystick.yaml')],
        output='screen',
    )

    joy = Node(
        package='joy_linux',
        executable='joy_linux_node',
        name='joy_linux',
        namespace=namespace,
        condition=IfCondition(use_joy),
        parameters=[
            os.path.join(teleop_share, 'config', 'joystick.yaml'),
            {'dev': joy_device},
        ],
        output='screen',
    )

    return LaunchDescription(
        args + [
            servo_driver,
            configure_servos,
            activate_servos,
            motion,
            expression,
            teleop,
            joy,
        ]
    )
