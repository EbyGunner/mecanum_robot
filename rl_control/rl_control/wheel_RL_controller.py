import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PoseStamped
import os, sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
EXT_LIB_PATH = os.path.join(SRC_DIR, 'mecanum_robot', 'rl_control', 'external_libraries')
sys.path.insert(0, EXT_LIB_PATH)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
import math

from . import wheel_velocity_translator
from . import distance_difference_calculator

class MecanumRLController(Node, gym.Env):
    def __init__(self):
        # Initialize both parent classes
        super().__init__('mecanum_rl_controller')
        gym.Env.__init__(self)
        
        # Mecanum wheel configuration (order: front_left, front_right, rear_left, rear_right)
        self.wheel_vel_pub = self.create_publisher(Float64MultiArray, '/wheel_velocity_controller/commands', 10)
        
        # Define observation space (lidar + position + orientation + goal info)
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(low=0, high=30, shape=(360,), dtype=np.float32),
            "position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "orientation": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "goal_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "goal_distance": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })


        
        # Action space: individual wheel velocities (rad/s)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Scale factor for converting normalized actions to real velocities
        self.action_scale = 10.0  # Converts [-1, 1] to [-10, 10] rad/s
        
        # Robot state variables
        self.current_position = np.zeros(2)
        self.current_orientation = np.zeros(1)
        self.goal_position = np.zeros(2)
        self.lidar_data = np.zeros(360)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        self.get_logger().info(f"Subscribed to lidar topic: {self.lidar_sub.topic_name}")
        self.get_logger().info(f"Subscribed to goal topic: {self.goal_sub.topic_name}")

        self.goal_position = None  # Initialize as None to detect when first goal arrives

        # RL parameters
        self.episode_length = 1000  # Max steps per episode
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.success_threshold = 0.1  # meters
        
        # Mecanum wheel parameters
        self.wheel_radius = 0.05  # meters (adjust according to your robot)
        self.robot_width = 0.3  # meters (distance between left/right wheels)
        self.robot_length = 0.3  # meters (distance between front/rear wheels)
        
        # Initialize RL model
        self.model = self._init_model()
        
        # Control timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Mecanum RL Controller initialized")

    def _init_model(self):
        """Initialize the RL model with appropriate architecture"""
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # Note: dict not list
        )

        return PPO(
            "MultiInputPolicy",
            self,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="auto"
        )
    
    def odom_callback(self, msg):
        """Callback for robot odometry"""
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y
        
        # Convert quaternion to yaw angle
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_orientation[0] = math.atan2(siny_cosp, cosy_cosp)
    
    def goal_callback(self, msg):
        """Enhanced goal callback with debugging"""
        self.goal_position = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(
            f"New goal received at: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})",
            throttle_duration_sec=1.0
        )
    
    def lidar_callback(self, msg):
        """Process lidar data to 360 elements"""
        ranges = np.array(msg.ranges)
        if len(ranges) > 360:
            step = len(ranges) // 360
            self.lidar_data = ranges[::step][:360]
        else:
            self.lidar_data = ranges
        
        # Replace inf/nan values with max range
        self.lidar_data = np.nan_to_num(self.lidar_data, posinf=30.0, neginf=0.0)
    
    def get_obs(self):
        """Get flattened observation dictionary"""
        if self.goal_position is None:
            self.get_logger().warn("No goal position set! Using zero position")
            self.goal_position = np.zeros(2)
        
        return {
            "lidar": self.lidar_data,
            "position": self.current_position,
            "orientation": self.current_orientation,
            "goal_position": self.goal_position,
            "goal_distance": np.array([np.linalg.norm(self.current_position - self.goal_position)])
        }
    
    
    def compute_reward(self) -> float:
        """Compute reward based on current state"""
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        
        # Calculate angle to goal
        angle_to_goal = math.atan2(
            self.goal_position[1] - self.current_position[1],
            self.goal_position[0] - self.current_position[0]
        )
        angle_diff = abs(self.current_orientation[0] - angle_to_goal)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Wrap around
        
        # Base reward for reducing distance
        distance_reward = -distance_to_goal * 0.5
        
        # Alignment reward for facing the goal
        alignment_reward = -angle_diff * 0.1
        
        # Obstacle penalty
        min_lidar = np.min(self.lidar_data)
        obstacle_penalty = 0.0
        if min_lidar < 0.3:  # Very close to obstacle
            obstacle_penalty = -10.0
        elif min_lidar < 0.5:  # Moderately close
            obstacle_penalty = -5.0
        
        # Success bonus
        success_bonus = 0.0
        if distance_to_goal < self.success_threshold:
            success_bonus = 100.0
            self.done = True
        
        # Time penalty to encourage faster navigation
        time_penalty = -0.01
        
        total_reward = (
            distance_reward + 
            alignment_reward + 
            obstacle_penalty + 
            success_bonus + 
            time_penalty
        )
        
        return total_reward
    
    def step(self, action: np.ndarray):
        """Execute one step with simplified observation"""
        # Scale and publish action
        real_action = action * self.action_scale
        vel_msg = Float64MultiArray()
        vel_msg.data = real_action.tolist()
        self.wheel_vel_pub.publish(vel_msg)
        
        # Get observation
        obs = self.get_obs()
        
        # Compute reward
        reward = self.compute_reward()
        self.episode_reward += reward
        
        self.current_step += 1
        if self.current_step >= self.episode_length:
            self.done = True
            
        info = {
            "distance_to_goal": obs["goal_distance"][0],
            "episode_reward": self.episode_reward
        }
        
        return obs, reward, self.done, info
    
    def reset(self, seed=None, options=None):
        """Reset environment with simplified observation"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        
        return self.get_obs(), {}
    
    def control_loop(self):
        """Main control loop that runs the RL agent"""

        if self.goal_position is None:
            self.get_logger().warn("Waiting for first goal position...", throttle_duration_sec=1.0)
            return
    
        if not self.done:
            # Get current observation
            obs = self.get_obs()
            
            # Get action from the model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute the action
            self.step(action)
        else:
            # Episode ended, log results and reset
            self.get_logger().info(
                f"Episode ended. Total reward: {self.episode_reward:.2f}, "
                f"Final distance: {np.linalg.norm(self.current_position - self.goal_position):.2f}m"
            )
            self.reset()
    
    def train(self, total_timesteps: int = 100000):
        """Train the RL model"""
        self.get_logger().info("Starting training...")
        
        # Verify the environment
        check_env(self)
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save the trained model
        self.model.save("ppo_mecanum_controller")
        self.get_logger().info("Training completed and model saved")
        
        # Evaluate the trained model
        self.evaluate()
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate the trained model"""
        self.get_logger().info(f"Evaluating model over {num_episodes} episodes...")
        
        total_rewards = []
        success_rate = 0
        
        for _ in range(num_episodes):
            obs = self.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            if info["distance_to_goal"] < self.success_threshold:
                success_rate += 1
            
            self.get_logger().info(
                f"Episode reward: {episode_reward:.2f}, "
                f"Final distance: {info['distance_to_goal']:.2f}m"
            )
        
        avg_reward = np.mean(total_rewards)
        success_rate = (success_rate / num_episodes) * 100
        
        self.get_logger().info(
            f"Evaluation complete. Average reward: {avg_reward:.2f}, "
            f"Success rate: {success_rate:.1f}%"
        )


    def test_open_loop_control(self):
        """Temporary test method to verify wheel control"""
        test_action = np.array([1.0, 1.0, 1.0, 1.0])  # Forward motion
        vel_msg = Float64MultiArray()
        vel_msg.data = test_action.tolist()
        self.wheel_vel_pub.publish(vel_msg)
        self.get_logger().info("Published test wheel velocities")

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    
    try:
        # Create nodes
        controller = MecanumRLController()
        translator = wheel_velocity_translator.MecanumWheelController()
        distance_calc = distance_difference_calculator.GoalDistanceCalculator()
        
        # Add nodes to executor
        executor.add_node(controller)
        executor.add_node(translator)
        executor.add_node(distance_calc)
        
        # Test without training first
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        controller.get_logger().error(f"Error: {str(e)}")
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()