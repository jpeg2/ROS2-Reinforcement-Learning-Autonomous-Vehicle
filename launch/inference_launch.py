"""
Launch file for running ML inference with gym bridge.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix
import os


def generate_launch_description():
    # Launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='./models/ppo_f1tenth_final.zip',
        description='Path to trained model'
    )

    # Resolve installed script path (in bin)
    pkg_prefix = get_package_prefix('car_vroom_vroom')
    ml_inference_exe = os.path.join(pkg_prefix, 'bin', 'ml_inference')

    # Execute the inference process directly
    ml_inference_proc = ExecuteProcess(
        cmd=[
            ml_inference_exe,
            '--ros-args',
            '-p', ['model_path:=', LaunchConfiguration('model_path')],
            '-p', 'scan_topic:=/ego_racecar/scan',
            '-p', 'odom_topic:=/ego_racecar/odom',
            '-p', 'drive_topic:=/ego_racecar/drive',
            '-p', 'max_range:=30.0',
            '-p', 'device:=auto',
        ],
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        ml_inference_proc,
    ])

