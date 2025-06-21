from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    RL_node = Node(
        package='rl_control',  
        executable='rl_controller',  
        name='rl_controller_node',
        output='screen'
    )


    return LaunchDescription([
        RL_node,
    ])