import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


def generate_launch_description():
    config_file = os.path.join(
            get_package_share_directory('image_detector_pkg'),
            'config',
            'config.yaml'
            )

    return LaunchDescription(
            Node(
                package='image_detector_pkg',
                executable='image_detector_node',
                name='image_detector_node',
                parameters=[config_file]
                )
            )
