import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import os, sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
EXT_LIB_PATH = os.path.join(SRC_DIR, 'mecanum_robot', 'rl_control', 'external_libraries')
sys.path.insert(0, EXT_LIB_PATH)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn
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
        self.action_scale = 30.0  # Converts [-1, 1] to [-10, 10] rad/s
        
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

        self.previous_distance = None

        # RL parameters
        self.episode_length = 1000  # Max steps per episode
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.success_threshold = 0.1  # meters

        # Progress monitoring parameters
        self.last_positions = []  # Store recent positions for progress tracking
        self.position_history_size = 15  # Number of positions to track
        self.min_position_change = 0.03  # Minimum position change (meters) to consider as progress
        self.speed_boost_factor = 1.5  # Multiplier when progress is too slow
        self.normal_speed = 20.0  # Normal maximum speed (rad/s)
        self.boosted_speed = 30.0  # Boosted maximum speed (rad/s)
        self.current_max_speed = self.normal_speed

        # Enhanced action scaling parameters
        self.max_initial_speed = 30.0  # rad/s (higher initial max)
        self.min_speed = 5.0          # rad/s (minimum when close to goal)
        self.distance_threshold = 1.0  # meters (start reducing speed within this distance)

        self.boost_active = False
        self.boost_counter = 0
        self.boost_duration = 20  # Number of steps to maintain boost
        self.boost_multiplier = 2.0  # How much to multiply speed when boosted
        
        # Initialize RL model
        self.model = self._init_model()
        
        # Control timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Enhanced recovery parameters
        self.off_course_threshold = math.pi/16  # Reduced to 45 degrees
        self.recovery_duration = 0
        self.max_recovery_steps = 20  # Increased
        self.recovery_cooldown = 0
        self.min_recovery_cooldown = 10  # Steps before another recovery can trigger
        self.collision_threshold = 0.3  # meters (lidar distance)
        self.last_recovery_type = "none"  # Initialize the tracking variable
        self.collision_detected = False
        self.collision_recovery_steps = 0
        self.max_collision_recovery = 15  # Steps for collision recovery
        
        self.last_turn_direction = 0  # Track turn direction between steps
        self.turn_persistence = 0  # How many steps we've been turning
        self.max_turn_persistence = 15  # Max steps to persist in one turn

        # Enhanced recovery parameters
        self.collision_zones = {
            'front': slice(0, 30),
            'front_left': slice(30, 60),
            'front_right': slice(-60, -30),
            'left': slice(60, 120),
            'right': slice(-120, -60)
        }
        self.recovery_sequence = [
                {'type': 'rotate', 'duration': 5, 'speed': 0.4},
                {'type': 'backup', 'duration': 5, 'speed': -0.3},
                {'type': 'strafe', 'duration': 10, 'speed': 0.5}
            ]
        self.max_recovery_attempts = 3  # Max recovery attempts before reset

        # Enhanced collision memory
        self.collision_memory = []
        self.max_collision_memory = 5  # Remember last 5 collisions
        self.collision_avoidance_duration = 30  # Steps to avoid collision-causing actions

        self.oscillation_counter = 0  # Track how many times we've oscillated
        self.max_oscillations = 5     # Max oscillations before forced recovery
        self.last_actions = []        # Store recent actions
        self.action_history_size = 10 # How many actions to remember
        self.blocked_directions = []  # Directions that led to collisions
                
        # Adaptive motion parameters
        self.safe_direction = None
        self.safe_direction_steps = 0
        
        self.get_logger().info("Mecanum RL Controller initialized")

    def _init_model(self):
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 128], vf=[256, 128]),  # Smaller network
            ortho_init=True,
            log_std_init=-1.0  # Encourage more exploration initially
        )
        
        return PPO(
            "MultiInputPolicy",
            self,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=2.5e-4,  # Slightly lower
            n_steps=1024,  # Smaller batch
            batch_size=32,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.15,  # Tighter clipping
            max_grad_norm=0.5,
            ent_coef=0.02
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
        new_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        
        # Only reset if goal has changed significantly
        if self.goal_position is None or np.linalg.norm(new_goal - self.goal_position) > 0.5:  # 0.5m threshold
            self.goal_position = new_goal
            self.get_logger().info(
                f"New goal received at: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})",
                throttle_duration_sec=1.0
            )
            self.reset()
        else:
            # Optional: Update goal position without resetting
            self.goal_position = new_goal
    
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
        
        # Add more nuanced direction reward
        angle_to_goal = math.atan2(self.goal_position[1] - self.current_position[1],
                                self.goal_position[0] - self.current_position[0])
        angle_diff = abs((self.current_orientation[0] - angle_to_goal + math.pi) % (2*math.pi) - math.pi)
        
        # Enhanced direction reward using exponential decay
        direction_reward = 0.5 * math.exp(-angle_diff)
        
        # Progressive distance reward
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        if self.previous_distance is not None:
            distance_change = self.previous_distance - distance_to_goal
            distance_reward = 1.0 * distance_change if distance_change > 0 else 2.0 * distance_change
        else:
            distance_reward = 0.0

        
        # Calculate movement direction relative to goal
        movement_direction_reward = 0.0
        if len(self.last_positions) > 1:
            movement_vector = self.current_position - self.last_positions[-1]
            if np.linalg.norm(movement_vector) > 0.01:  # Only if actually moved
                goal_vector = self.goal_position - self.current_position
                if np.linalg.norm(goal_vector) > 0.01:  # Only if not at goal
                    cos_theta = np.dot(movement_vector, goal_vector) / (
                        np.linalg.norm(movement_vector) * np.linalg.norm(goal_vector))
                    movement_direction_reward = 0.3 * cos_theta  # Scale factor
        
        # Enhanced distance reduction reward (now with direction sensitivity)
        distance_reduction_reward = 0.0
        if self.previous_distance is not None:
            distance_change = self.previous_distance - distance_to_goal
            if distance_change > 0:  # Getting closer
                distance_reduction_reward = distance_change * 0.7  # Increased from 0.5
            else:  # Getting farther
                distance_reduction_reward = distance_change * 1.2  # Stronger penalty
        
        # Update previous distance
        self.previous_distance = distance_to_goal
        
            # Enhanced orientation penalty
        angle_to_goal = math.atan2(
            self.goal_position[1] - self.current_position[1],
            self.goal_position[0] - self.current_position[0]
        )

        # Modify the alignment penalty to be direction-aware
        angle_error = (angle_to_goal - self.current_orientation[0]) % (2*math.pi)
        if angle_error > math.pi:
            angle_error = angle_error - 2*math.pi

        # Use a smoother penalty curve
        alignment_penalty = -0.2 * math.tanh(2 * angle_error)  # Smoother, less aggressive penalty
        
        # Speed reward (penalize fast movement in wrong direction)
        current_speed = 0.0
        if len(self.last_positions) > 1:
            current_speed = np.linalg.norm(self.last_positions[-1] - self.last_positions[-2]) / 0.1
        
        speed_reward = np.sqrt(current_speed) * (0.5 if distance_reduction_reward >=0 else 0.2)
        
        # Obstacle penalty
        min_lidar = np.min(self.lidar_data)
        obstacle_penalty = 0.0
        if min_lidar < 0.5:
            obstacle_penalty = -10.0 * (0.5 - min_lidar)
        
        # Success bonus
        success_bonus = 100.0 if distance_to_goal < self.success_threshold else 0.0
        
        # Time penalty (reduced when moving correctly)
        time_penalty = -0.002 if distance_reduction_reward >=0 else -0.005
        
        total_reward = (
            movement_direction_reward +
            distance_reduction_reward +
            speed_reward +
            alignment_penalty + 
            obstacle_penalty + 
            success_bonus + 
            time_penalty
        )
        
        # Debug logging
        self.get_logger().info(
            f"Reward components: "
            f"Dir={movement_direction_reward:.2f} "
            f"DistChg={distance_reduction_reward:.2f} "
            f"Speed={speed_reward:.2f} "
            f"Align={alignment_penalty:.2f} "
            f"Obs={obstacle_penalty:.2f}",
            throttle_duration_sec=1.0
        )
        
        return total_reward
    
    def step(self, action: np.ndarray):
        """Dynamically scale actions with progress monitoring"""
        # Store current position for progress tracking
        self.last_positions.append(self.current_position.copy())
        if len(self.last_positions) > self.position_history_size:
            self.last_positions.pop(0)

        # Store action history
        self.last_actions.append(action.copy())
        if len(self.last_actions) > self.action_history_size:
            self.last_actions.pop(0)

        # Detect oscillation pattern
        if len(self.last_actions) == self.action_history_size:
            action_changes = np.sum(np.abs(np.diff(self.last_actions, axis=0)))
            if action_changes > 5.0:  # High change = oscillating
                self.oscillation_counter += 1
            else:
                self.oscillation_counter = max(0, self.oscillation_counter - 1)

        if self.oscillation_counter > self.max_oscillations:
            recovery_action = self._generate_escape_pattern()
            blend_factor = 1.0  # Full override
            action = recovery_action
            self.get_logger().warn("FORCED RECOVERY DUE TO OSCILLATION", throttle_duration_sec=1.0)
            self.oscillation_counter = 0
        
        # Calculate progress and angles
        progress = 0.0
        if len(self.last_positions) == self.position_history_size:
            progress = np.linalg.norm(self.last_positions[-1] - self.last_positions[0])
        
        angle_to_goal = math.atan2(
            self.goal_position[1] - self.current_position[1],
            self.goal_position[0] - self.current_position[0]
        )
        angle_diff = abs(self.current_orientation[0] - angle_to_goal)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        
        # --- Enhanced Recovery System ---
        collision_info = self._check_collision()
        recovery_active = False
        recovery_action = np.zeros(4)
        recovery_type = "none"
        
        # 1. Handle collisions
        if collision_info is not None:
            # Store collision in memory
            if len(self.collision_memory) >= self.max_collision_memory:
                self.collision_memory.pop(0)
            self.collision_memory.append({
                'position': self.current_position.copy(),
                'orientation': self.current_orientation[0],
                'action': action.copy(),
                'info': collision_info
            })
            
            # Determine safe direction (opposite to collision angle)
            self.safe_direction = (collision_info['angle'] + math.pi) % (2*math.pi)
            self.safe_direction_steps = self.collision_avoidance_duration
            
            # Emergency recovery sequence
            recovery_type = "emergency"
            recovery_active = True
            self.collision_recovery_steps += 1
            
            # Staged recovery:
            if self.collision_recovery_steps < 10:  # Back up
                recovery_action = np.array([-0.8, -0.8, -0.8, -0.8])
            elif self.collision_recovery_steps < 25:  # Turn to safe direction
                angle_to_safe = (self.safe_direction - self.current_orientation[0]) % (2*math.pi)
                turn_dir = 1 if angle_to_safe < math.pi else -1
                recovery_action = np.array([
                    -0.6 * turn_dir, 0.6 * turn_dir, -0.6 * turn_dir, 0.6 * turn_dir
                ])
            else:  # Move in safe direction
                recovery_action = np.array([0.5, 0.5, 0.5, 0.5])  # Forward
                
            if self.collision_recovery_steps > 40:
                self.collision_recovery_steps = 0
                self.recovery_cooldown = 20

        # 2. Avoid repeating collision-causing actions
        elif self.safe_direction_steps > 0:
            self.safe_direction_steps -= 1
            
            # Modify action to favor safe direction
            angle_to_safe = (self.safe_direction - self.current_orientation[0]) % (2*math.pi)
            safe_factor = max(0, 1 - (angle_to_safe / math.pi))  # 1 when aligned, 0 when opposite
            
            # Reduce action components that would move toward danger
            action = action * (0.3 + 0.7 * safe_factor)
            
            # Add slight bias toward safe direction
            turn_bias = 0.2 * (1 if angle_to_safe < math.pi else -1)
            action += np.array([-turn_bias, turn_bias, -turn_bias, turn_bias])

        # 3. Course correction (when no collision concerns)
        elif (not recovery_active and 
            self.recovery_cooldown <= 0 and
            angle_diff > self.off_course_threshold):
            
            self.recovery_duration += 1
            turn_dir = 1 if (angle_to_goal - self.current_orientation[0]) % (2*math.pi) < math.pi else -1
            
            # Progressive correction with persistence
            if self.last_recovery_type != "course_correction":
                self.turn_persistence = 0
                self.last_turn_direction = turn_dir
            
            # Maintain turn direction for minimum time
            if self.turn_persistence < self.max_turn_persistence:
                turn_dir = self.last_turn_direction
                self.turn_persistence += 1
            
            rotational_component = 0.6 * turn_dir
            forward_component = 0.4 if angle_diff < math.pi/2 else 0.1
            
            recovery_action = np.array([
                -rotational_component + forward_component,
                rotational_component + forward_component,
                -rotational_component + forward_component,
                rotational_component + forward_component
            ])
            
            recovery_active = True
            recovery_type = "course_correction"
            
            # Only exit when properly aligned
            if angle_diff < math.pi/6:  # 30 degrees
                self.recovery_duration = 0
                self.recovery_cooldown = 10

        # 4. Stuck recovery (lowest priority)
        elif (not recovery_active and
            self.recovery_cooldown <= 0 and
            self._calculate_movement_metrics()['stuck']):
            
            self.recovery_duration += 1
            pattern = self.recovery_duration % 6  # Longer pattern sequence
            
            if pattern == 0:
                recovery_action = np.array([0.5, -0.5, 0.5, -0.5])  # Right strafe
            elif pattern == 1:
                recovery_action = np.array([-0.5, 0.5, -0.5, 0.5])  # Left strafe
            elif pattern == 2:
                recovery_action = np.array([0.7, 0.7, 0.7, 0.7])   # Forward
            elif pattern == 3:
                recovery_action = np.array([-0.4, -0.4, -0.4, -0.4]) # Backward
            else:  # Random wiggle
                recovery_action = np.random.uniform(-0.3, 0.3, 4)
                
            recovery_active = True
            recovery_type = "stuck_recovery"
            
            if self.recovery_duration > 50:
                self.recovery_duration = 0
                self.recovery_cooldown = 20

        # Apply recovery blending
        if recovery_active:
            # More aggressive override when in collision recovery
            if recovery_type == "emergency":
                blend_factor = 1.0  # Full override
            else:
                blend_factor = min(1.0, 0.7 + 0.3 * (self.recovery_duration / 10))
            
            action = action * (1 - blend_factor) + recovery_action * blend_factor
            
            # Reset oscillation counter when in recovery
            self.oscillation_counter = 0
            
            # Log recovery state changes
            if recovery_type != self.last_recovery_type:
                log_msg = f"Recovery: {recovery_type} | Angle: {angle_diff:.2f}rad | Steps: {self.recovery_duration}"
                if recovery_type == "emergency":
                    log_msg += f" | Safe Dir: {math.degrees(self.safe_direction):.1f}°"
                self.get_logger().info(log_msg, throttle_duration_sec=0.5)
        
        # --- Speed Control ---
        if self.boost_active:
            self.boost_counter += 1
            if self.boost_counter > 20:
                self.boost_active = False
        elif (progress < self.min_position_change and 
                len(self.last_positions) == self.position_history_size and 
                np.min(self.lidar_data) > 0.8 and 
                not recovery_active):  # Don't boost during recovery
            self.boost_active = True
            self.boost_counter = 0
            self.get_logger().info("ACTIVATING SPEED BOOST", throttle_duration_sec=1.0)
        
        current_max_speed = self.boosted_speed if self.boost_active else self.normal_speed
        
        # Dynamic speed scaling based on distance to goal
        distance = np.linalg.norm(self.current_position - self.goal_position)
        if distance < self.distance_threshold:
            t = min(1.0, distance / self.distance_threshold)
            speed_scale = self.min_speed + (current_max_speed - self.min_speed) * (1 - (1-t)**2)
            current_max_speed = np.clip(speed_scale, self.min_speed, current_max_speed)
        
        # Scale final action
        real_action = np.tanh(action) * current_max_speed

        # Publish and return
        vel_msg = Float64MultiArray()
        vel_msg.data = real_action.tolist()
        self.wheel_vel_pub.publish(vel_msg)
        
        obs = self.get_obs()
        reward = self.compute_reward()
        self.episode_reward += reward
        self.current_step += 1
        
        # Termination conditions
        self.done = (
            self.current_step >= self.episode_length or
            (self.previous_distance is not None and 
            obs["goal_distance"][0] > self.previous_distance * 2.0 and 
            self.current_step > 30) or
            (recovery_active and 
            self.recovery_duration > 100) or
            self.oscillation_counter > self.max_oscillations * 2
        )
        
        info = {
            "distance_to_goal": obs["goal_distance"][0],
            "episode_reward": self.episode_reward,
            "recovery_active": recovery_active
        }
        
        return obs, reward, self.done, info
    

    def _generate_escape_pattern(self):
        """Generate an escape pattern based on collision history"""
        if not self.blocked_directions:
            # No history - try random strafe
            return np.random.choice([np.array([0.5, -0.5, 0.5, -0.5]), 
                                np.array([-0.5, 0.5, -0.5, 0.5])])
        
        # Try to move away from most common collision direction
        avg_blocked_dir = np.mean(self.blocked_directions)
        turn_dir = 1 if (avg_blocked_dir - self.current_orientation[0]) % (2*math.pi) < math.pi else -1
        
        return np.array([
            -0.6 * turn_dir, 0.6 * turn_dir, -0.6 * turn_dir, 0.6 * turn_dir
        ])
    

    def _check_collision(self):
        """Enhanced collision detection that tracks problematic directions"""
        front_zone = np.concatenate([self.lidar_data[:30], self.lidar_data[-30:]])
        front_min = np.min(front_zone)
        
        if front_min < self.collision_threshold:
            # Find direction of closest obstacle
            min_idx = np.argmin(self.lidar_data)
            collision_angle = math.radians(min_idx)
            
            # Add to blocked directions
            if len(self.blocked_directions) > 5:
                self.blocked_directions.pop(0)
            self.blocked_directions.append(collision_angle)
            
            return {
                'distance': front_min,
                'angle': collision_angle
            }
        return None
    

    def _calculate_movement_metrics(self):
        """Calculate various movement metrics"""
        metrics = {
            'distance': 0.0,
            'stuck': False
        }
        
        if len(self.last_positions) >= 2:
            # Calculate recent movement
            metrics['distance'] = np.linalg.norm(
                self.last_positions[-1] - self.last_positions[0]
            )
            
            # Check if stuck (no meaningful movement)
            if (len(self.last_positions) == self.position_history_size and
                metrics['distance'] < self.min_position_change and
                np.linalg.norm(self.current_position - self.goal_position) > 0.5):
                metrics['stuck'] = True
                
        return metrics
    
    def reset(self, seed=None, options=None):
        """Reset environment and progress tracking"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.last_positions = []
        self.current_max_speed = self.normal_speed
        self.previous_distance = None
        
        # Reset recovery states
        self.recovery_duration = 0
        self.recovery_cooldown = 0
        self.collision_detected = False
        self.collision_recovery_steps = 0
        self.last_recovery_type = "none"
        
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

        eval_callback = EvalCallback(
            self,
            best_model_save_path=os.path.join(SRC_DIR, 'mecanum_robot', 'rl_control', 'train', 'model'),
            log_path=os.path.join(SRC_DIR, 'mecanum_robot', 'rl_control', 'train', 'logs'),
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
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