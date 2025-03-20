import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener
import math

class GoalDistanceCalculator(Node):
    def __init__(self):
        super().__init__('goal_distance_calculator')

        # TF Buffer and Listener to get the robot's current pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to goal pose from RViz
        self.goal_sub = self.create_subscription(
            PoseStamped, 
            '/goal_pose', 
            self.goal_callback, 
            10
        )

        self.goal_pose = None  # Store the latest goal
        self.timer = self.create_timer(0.5, self.calculate_distance)  # Update every 0.5s

    def goal_callback(self, msg):
        """Update goal pose when a new one is received from RViz."""
        self.goal_pose = msg
        self.get_logger().info("New goal received!")

    def get_robot_pose(self):
        """Get the robot's current pose from SLAM (base_link in map frame)."""
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'map',  # SLAM frame
                'base_link',  # Robot's base frame
                rclpy.time.Time()
            )
            return transform.transform
        except:
            self.get_logger().warn("Waiting for transform from map → base_link")
            return None

    def calculate_distance(self):
        """Compute the distance and orientation difference between robot and goal."""
        if self.goal_pose is None:
            return
        
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        # Extract robot's position
        rx, ry = robot_pose.translation.x, robot_pose.translation.y

        # Extract goal's position
        gx, gy = self.goal_pose.pose.position.x, self.goal_pose.pose.position.y

        # Calculate Euclidean distance
        distance = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)

        # Extract robot's and goal's yaw (from quaternion to yaw)
        r_yaw = self.quaternion_to_yaw(robot_pose.rotation)
        g_yaw = self.quaternion_to_yaw(self.goal_pose.pose.orientation)

        # Compute orientation difference (ensure it's between -π to π)
        yaw_diff = math.atan2(math.sin(g_yaw - r_yaw), math.cos(g_yaw - r_yaw))

        self.get_logger().info(f"Distance to Goal: {distance:.2f} meters, Orientation Diff: {math.degrees(yaw_diff):.2f}°")

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle in radians."""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))