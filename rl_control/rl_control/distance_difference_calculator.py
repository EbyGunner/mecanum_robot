#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener
from message_interfaces.msg import GoalCurrentPose
import math

class GoalCurrentPosePublisher(Node):
    def __init__(self):
        super().__init__('goal_current_pose_publisher')

        # TF Buffer and Listener to get the robot's current pose (map -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to goal pose (e.g. from RViz)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.goal_pose: PoseStamped | None = None
        self.latest_transform: TransformStamped | None = None

        # Publisher for the combined message
        self.pub = self.create_publisher(
            GoalCurrentPose,
            'goal_current_pose',
            10
        )

        # Timer at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Distance tracking
        self.last_distance_log_time = self.get_clock().now()
        self.distance_log_interval = 2.0  # Log every 2 seconds to avoid spam

    def goal_callback(self, msg: PoseStamped):
        """Store latest goal pose as received (keeps original PoseStamped)."""
        self.goal_pose = msg
        self.get_logger().info('Received new goal pose.')

    def lookup_robot_transform(self) -> TransformStamped | None:
        """Try to lookup transform map -> base_link. Returns TransformStamped or None."""
        try:
            # use latest available transform
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'map',        # target frame
                'base_link',  # source frame
                rclpy.time.Time()  # latest
            )
            return transform
        except Exception as e:
            # quiet warning, don't spam too often
            self.get_logger().debug(f"Transform lookup failed: {e}")
            return None

    def calculate_distance(self, goal_pose: PoseStamped, current_transform: TransformStamped) -> float:
        """Calculate Euclidean distance between goal and current position."""
        goal_x = goal_pose.pose.position.x
        goal_y = goal_pose.pose.position.y
        
        current_x = current_transform.transform.translation.x
        current_y = current_transform.transform.translation.y
        
        dx = goal_x - current_x
        dy = goal_y - current_y
        
        return math.sqrt(dx*dx + dy*dy)

    def timer_cb(self):
        """Publish combined message when both the goal and current transform are available."""
        if self.goal_pose is None:
            # no goal yet
            return

        transform = self.lookup_robot_transform()
        if transform is None:
            return

        # assemble combined message (keeps original message types)
        combined = GoalCurrentPose()
        # assign copies (these are full ROS message objects)
        combined.goal = self.goal_pose
        combined.current_transform = transform

        # publish
        self.pub.publish(combined)

        # Calculate and log distance (throttled to avoid spam)
        current_time = self.get_clock().now()
        time_since_last_log = (current_time - self.last_distance_log_time).nanoseconds / 1e9
        
        if time_since_last_log >= self.distance_log_interval:
            distance = self.calculate_distance(self.goal_pose, transform)
            
            goal_x = self.goal_pose.pose.position.x
            goal_y = self.goal_pose.pose.position.y
            current_x = transform.transform.translation.x
            current_y = transform.transform.translation.y
            
            self.get_logger().info(
                f"Distance to goal: {distance:.3f}m | "
                f"Goal: ({goal_x:.2f}, {goal_y:.2f}) | "
                f"Current: ({current_x:.2f}, {current_y:.2f})",
                throttle_duration_sec=1.0
            )
            
            self.last_distance_log_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = GoalCurrentPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()