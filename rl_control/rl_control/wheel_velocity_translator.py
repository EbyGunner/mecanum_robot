from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np
import rclpy

class MecanumWheelController(Node):
    def __init__(self):
        super().__init__('mecanum_wheel_controller')
        
        # QoS configuration
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriber and publisher
        self.wheel_sub = self.create_subscription(
            Float64MultiArray,
            '/wheel_velocity_controller/commands',
            self.wheel_callback,
            qos
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        
        # Robot parameters (adjust these to match your actual robot)
        self.wheel_radius = 0.029  # meters
        self.robot_width = 0.226   # meters (distance between left/right wheels)
        self.robot_length = 0.235  # meters (distance between front/rear wheels)
        
        # Velocity limits
        self.max_wheel_velocity = 10.0  # rad/s
        self.max_linear_velocity = 1.0  # m/s
        self.max_angular_velocity = 2.0  # rad/s
        
        # Compute forward kinematics matrix (converts wheel velocities to body twist)
        L = self.robot_length
        W = self.robot_width
        R = self.wheel_radius
        self.forward_kinematics = np.array([
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [-1/(L+W), 1/(L+W), -1/(L+W), 1/(L+W)]
        ]) * (R/4)
        
        self.get_logger().info(f"Forward kinematics matrix:\n{self.forward_kinematics}")
        
        # Test publisher
        self.timer = self.create_timer(1.0, self.test_publish)

    def wheel_callback(self, msg):
        """Convert wheel velocities to body twist"""
        try:
            # Validate input
            if len(msg.data) != 4:
                raise ValueError(f"Expected 4 wheel velocities, got {len(msg.data)}")
            
            # Convert to numpy array and reshape
            wheel_vel = np.array(msg.data, dtype=np.float64).reshape(4, 1)
            
            # Compute body velocities (matrix multiplication)
            body_vel = self.forward_kinematics @ wheel_vel
            
            # Create and publish Twist message
            twist = Twist()
            twist.linear.x = float(np.clip(body_vel[0, 0], -self.max_linear_velocity, self.max_linear_velocity))
            twist.linear.y = float(np.clip(body_vel[1, 0], -self.max_linear_velocity, self.max_linear_velocity))
            twist.angular.z = float(np.clip(body_vel[2, 0], -self.max_angular_velocity, self.max_angular_velocity))
            
            self.cmd_vel_pub.publish(twist)
            
            self.get_logger().info(
                f"Converted wheel velocities {msg.data} to "
                f"twist: lin.x={twist.linear.x:.2f}, "
                f"lin.y={twist.linear.y:.2f}, "
                f"ang.z={twist.angular.z:.2f}",
                throttle_duration_sec=0.5
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in wheel callback: {str(e)}")

    def test_publish(self):
        """Test method to verify publisher is working"""
        twist = Twist()
        twist.linear.x = 0.1
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Published TEST cmd_vel message")