import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

import wheel_velocity_translator
import distance_difference_calculator

class RL_controller(Node):
    def __init__(self):
        self.start = True

    def wheel_callback(self):
        """Callback function to process wheel velocities and compute vehicle velocity"""
        self.callback = True

def main(args=None):
    rclpy.init(args=args)
    translator_node = wheel_velocity_translator.Mecanumwheeltobody()
    distance_node = distance_difference_calculator.GoalDistanceCalculator()
    rclpy.spin(translator_node)
    rclpy.spin(distance_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
