import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class Mecanumwheeltobody(Node):
    def __init__(self):
        super().__init__('mecanum_controller')

        # Subscriber to wheel velocity topic
        self.wheel_sub = self.create_subscription(
            Float64MultiArray,
            '/wheel_velocity_controller/commands',
            self.wheel_callback,
            10
        )

        # Publisher for vehicle velocity
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Robot parameters
        self.wheel_radius = 0.029  # Wheel radius (meters)
        self.robot_width = 0.226   # Distance between left & right wheels (meters)
        self.robot_length = 0.235  # Distance between front & rear wheels (meters)

    def wheel_callback(self, msg):
        """Callback function to process wheel velocities and compute vehicle velocity"""
        if len(msg.data) != 4:
            self.get_logger().error("Received incorrect wheel velocity data")
            return

        # Extract wheel velocities
        w1, w2, w3, w4 = msg.data  # [front_left, front_right, rear_left, rear_right]

        # Compute vehicle velocity using inverse kinematics
        L, W = self.robot_length, self.robot_width
        R = self.wheel_radius

        vx = (w1 + w2 + w3 + w4) * (R / 4)
        vy = (-w1 + w2 + w3 - w4) * (R / 4)
        wz = (-w1 + w2 - w3 + w4) * (R / (4 * (L + W)))

        # Create and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = vx
        twist_msg.linear.y = vy
        twist_msg.angular.z = wz

        self.cmd_vel_pub.publish(twist_msg)

        self.get_logger().info(f'Published cmd_vel: x={vx:.3f}, y={vy:.3f}, z={wz:.3f}')