import os
import xacro
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit


def generate_launch_description():

    # Declare RL usage argument
    RL_or_nav_selection = DeclareLaunchArgument(
        'use_RL',
        default_value='false',  # Boolean values should be given as strings
        description='Use RL control algorithm if true'
    )

    package_path = get_package_share_directory('mecanum_robot_simulation')
    bridge_params = os.path.join(package_path, 'src', 'config', 'gazeebo_ros_bridge.yaml')

    # Declare world argument
    arguments = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(package_path, 'src', 'world', 'mecanum_world'),
        description='Gz sim World'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments=[
            ('gz_args', [LaunchConfiguration('world'),
                         '.sdf',
                         ' -v 4',  # Verbose mode for logging
                         ' -r']  # Starts the simulation in real-time mode
             )
        ]
    )

    # Load URDF model
    xacro_file = os.path.join(package_path, 'src', 'description', 'urdf', 'urdf_final.urdf')
    doc = xacro.process_file(xacro_file, mappings={'use_sim': 'true'})
    robot_desc = doc.toprettyxml(indent='  ')

    params = {'robot_description': robot_desc}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-string', robot_desc,
                   '-x', '1.0',
                   '-y', '0.0',
                   '-z', '0.07',
                   '-R', '0.0',
                   '-P', '0.0',
                   '-Y', '0.0',
                   '-name', 'mecanum_robot',
                   '-allow_renaming', 'false'],
    )

    # Bridge
    bridge = Node(package='ros_gz_bridge',
                  executable='parameter_bridge',
                  arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
                  output='screen')

    gazebo_ros_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera/image_raw'],
        output='screen',
    )

    rviz_config_file = os.path.join(package_path, 'src', 'rviz', 'robot_view.rviz')

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
    )

    # SLAM argument
    slam_arg = DeclareLaunchArgument(
        'slam',
        default_value='True',
        description='Whether to run SLAM'
    )

    # Select navigation or RL based on the argument
    navigation_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package_path, 'src', 'launch', 'navigation_launch.py')),
        launch_arguments={'slam': LaunchConfiguration('slam')}.items()
    )

    RL_node = Node(
        package='rl_control',  
        executable='rl_controller',  
        name='RL algorithm launcher',
        output='screen'
    )

    # Conditional selection using opaque function
    from launch.actions import OpaqueFunction

    def select_navigation_or_RL(context):
        use_RL = context.launch_configurations.get('use_RL', 'false').lower() == 'true'
        return [RL_node] if use_RL else [navigation_node]

    selection_node = OpaqueFunction(function=select_navigation_or_RL)

    # SLAM node
    slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package_path, 'src', 'launch', 'online_async_launch.py'))
    )

    return LaunchDescription([
        RL_or_nav_selection,
        slam_arg,
        arguments,
        gazebo,
        node_robot_state_publisher,
        gz_spawn_entity,
        bridge,
        gazebo_ros_image_bridge,
        rviz_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[slam_node, selection_node],
            )
        ),
    ])
