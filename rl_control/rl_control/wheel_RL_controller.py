import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan

from . import wheel_velocity_translator
from . import distance_difference_calculator

import gymnasium as gym
import numpy as np

class RL_controller(Node, gym.Env):
    def __init__(self):

        super().__init__('rl_controller_node')

        self.cmd_vel_pub = self.create_publisher(Float64MultiArray, '/wheel_velocity_controller/commands', 10)

        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(362,), dtype=np.float32)

        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)

        # Subscriber to goal-position difference topic
        self.wheel_sub = self.create_subscription(
            Float64MultiArray,
            '/goal_position_difference',
            self.goal_difference,
            10
        )

        # Subscriber to lidar topic
        self.wheel_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )


    def goal_difference(self, msg):
        """Callback function to process positional difference between the current position and the goal"""
        self.position_diff, self.orientation_diff = msg

    def lidar_callback(self, msg):
        """Callback function to process lidar data"""
         

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
