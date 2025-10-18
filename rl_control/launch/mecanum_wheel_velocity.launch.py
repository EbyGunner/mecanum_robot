from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    RL_node = Node(
        package='rl_control',  
        executable='wheel_velocity',  
        name='wheel_velocity_calculator',
        output='screen'
    )


    return LaunchDescription([
        RL_node,
    ])