import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit


def generate_launch_description():

    # Declare RL usage argument
    RL_or_nav_selection = DeclareLaunchArgument(
        'use_RL',
        default_value='true',
        description='Use RL control algorithm if true'
    )

    package_path = get_package_share_directory('mecanum_robot_simulation')
    bridge_params = os.path.join(package_path, 'config', 'gazeebo_ros_bridge.yaml')

    rl_package = get_package_share_directory('rl_control')

    # Declare world argument
    arguments = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(package_path, 'world', 'mecanum_world'),
        description='Gz sim World'
    )

    # SLAM argument
    slam_arg = DeclareLaunchArgument(
        'slam',
        default_value='True',
        description='Whether to run SLAM'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments=[
            ('gz_args', [LaunchConfiguration('world'),
                         '.sdf',
                         ' -v 4',  # Verbose mode for logging
                         ' -r']    # Starts the simulation in real-time mode
             )
        ]
    )

    # Load URDF model
    xacro_file = os.path.join(package_path, 'urdf', 'urdf_final.urdf')
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

    rviz_config_file = os.path.join(package_path, 'rviz', 'robot_view.rviz')

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
    )

    # SLAM node
    slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package_path, 'launch', 'online_async_launch.py'))
    )

    # Define RL support nodes
    wheel_velocity_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(rl_package, 'launch', 'mecanum_wheel_velocity.launch.py')),
    )

    distance_difference_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(rl_package, 'launch', 'mecanum_distance_difference.launch.py')),
    )

    # Select navigation or RL based on the argument
    navigation_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package_path, 'launch', 'navigation_launch.py')),
        launch_arguments={'slam': LaunchConfiguration('slam')}.items()
    )

    RL_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(rl_package, 'launch', 'mecanum_rl_control.launch.py')),
    )

    def select_nodes(context):
        """Select which nodes to launch based on use_RL parameter"""
        use_RL = context.launch_configurations.get('use_RL', 'false').lower() == 'true'
        
        if use_RL:
            # Launch RL node with its support nodes
            return [RL_node, wheel_velocity_node, distance_difference_node]
        else:
            # Launch only navigation node
            return [navigation_node]

    selected_nodes = OpaqueFunction(function=select_nodes)

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
                on_exit=[slam_node],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[selected_nodes],
            )
        ),
    ])