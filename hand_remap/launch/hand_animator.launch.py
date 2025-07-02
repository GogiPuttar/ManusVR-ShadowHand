from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get the path to the config and data folders
    pkg_share = get_package_share_directory('hand_remap')
    config_dir = os.path.join(pkg_share, 'config')
    data_dir = os.path.join(pkg_share, 'data')

    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to launch RViz'
        ),
        Node(
            package='hand_remap',
            executable='hand_animator_node',
            name='hand_animator',
            parameters=[{'csv_path': os.path.join(data_dir, 'skeleton_log_20250624_203229.csv')}],
            output='screen'
        ),
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
