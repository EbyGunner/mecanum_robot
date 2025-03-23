import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from . import wheel_velocity_translator
from . import distance_difference_calculator

class RL_controller(Node):
    def __init__(self):

        super().__init__('rl_controller_node')

        self.cmd_vel_pub = self.create_publisher(Float64MultiArray, '/wheel_velocity_controller/commands', 10)

    def wheel_callback(self):
        """Callback function to process wheel velocities and compute vehicle velocity"""
        self.callback = True

def main(args=None):
    rclpy.init(args=args)
    
    translator_node = wheel_velocity_translator.Mecanumwheeltobody()
    distance_node = distance_difference_calculator.GoalDistanceCalculator()
    rl_controller_node = RL_controller()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(translator_node)
    executor.add_node(distance_node)
    executor.add_node(rl_controller_node)

    executor.spin()
    executor.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
