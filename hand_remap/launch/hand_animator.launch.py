from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get the path to the config and data folders
    pkg_share = get_package_share_directory('hand_remap')
    config_dir = os.path.join(pkg_share, 'config')
    data_dir = os.path.join(pkg_share, 'data')

    use_rviz = LaunchConfiguration('use_rviz')

    # Paths to hand urdf/xacro
    hand_xacro = PathJoinSubstitution([
        FindPackageShare('sr_description'),
        'robots', 'sr_hand.urdf.xacro'
    ])

    # Generate robot description using hand xacro
    hand_robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', hand_xacro]),
            value_type=str
        )
    }

    # Hand robot state publisher
    hand_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[hand_robot_description],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to launch RViz'
        ),
        DeclareLaunchArgument(
            'tracking_path',
            default_value='true',
            description='Settings for motion tracking'
        ),
        Node(
            package='hand_remap',
            executable='hand_animator_node',
            name='hand_animator',
            parameters=[{
                'csv_path': os.path.join(data_dir, 'skeleton_log_20250624_203229.csv'),
                'tracking_path': os.path.join(config_dir, 'tracking_params.yaml'),
                **hand_robot_description,
                }],
            output='screen'
        ),
        hand_rsp,
        # RVIZ
        Node(
            condition=IfCondition(use_rviz),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(config_dir, 'hand_animator.rviz')],
            output='screen'
        )
    ])
