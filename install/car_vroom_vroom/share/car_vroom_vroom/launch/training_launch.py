"""
Launch file for training the ML model.
This runs training independently of ROS2.
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.arguments import LaunchArgument
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share = get_package_share_directory('car_vroom_vroom')
    
    # Get training script path
    training_script = os.path.join(
        get_package_share_directory('car_vroom_vroom'),
        '..',
        '..',
        'src',
        'car_vroom_vroom',
        'car_vroom_vroom',
        'ml_train.py'
    )
    training_script = os.path.abspath(training_script)
    
    # Training process
    training_node = ExecuteProcess(
        cmd=[
            'python3',
            training_script,
            '--map-path', '/home/contranickted/sim_ws/src/f1tenth_gym_ros/maps/levine',
            '--total-timesteps', '1000000',
            '--n-envs', '4',
            '--device', 'auto',
        ],
        output='screen'
    )
    
    return LaunchDescription([
        training_node,
    ])

