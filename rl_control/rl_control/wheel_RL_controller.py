import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from message_interfaces.msg import GoalCurrentPose
import os
import sys
import math
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
EXT_LIB_PATH = os.path.join(SRC_DIR, 'mecanum_robot', 'rl_control', 'external_libraries')
sys.path.insert(0, EXT_LIB_PATH)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn

from . import wheel_velocity_translator
from . import distance_difference_calculator


class RewardCalculator:
    """Handles all reward calculation logic"""
    
    def __init__(self, success_threshold=0.1):
        self.success_threshold = success_threshold
        self.previous_distance = None
        
    def compute(self, current_position, goal_position, orientation, lidar_data, last_positions):
        """Compute total reward based on current state"""
        distance_to_goal = np.linalg.norm(current_position - goal_position)
        
        rewards = {
            'movement_direction': self._movement_direction_reward(current_position, goal_position, last_positions),
            'distance_reduction': self._distance_reduction_reward(distance_to_goal),
            'speed': self._speed_reward(last_positions),
            'alignment': self._alignment_penalty(current_position, goal_position, orientation),
            'obstacle': self._obstacle_penalty(lidar_data),
            'success': self._success_bonus(distance_to_goal),
            'time': self._time_penalty()
        }
        
        self.previous_distance = distance_to_goal
        return sum(rewards.values()), rewards
    
    def _movement_direction_reward(self, current_position, goal_position, last_positions):
        """Reward for moving in the right direction"""
        if len(last_positions) < 2:
            return 0.0
            
        movement_vector = current_position - last_positions[-1]
        if np.linalg.norm(movement_vector) <= 0.01:
            return 0.0
            
        goal_vector = goal_position - current_position
        if np.linalg.norm(goal_vector) <= 0.01:
            return 0.0
            
        cos_theta = np.dot(movement_vector, goal_vector) / (
            np.linalg.norm(movement_vector) * np.linalg.norm(goal_vector))
        return 0.3 * cos_theta
    
    def _distance_reduction_reward(self, current_distance):
        """Reward/penalty based on distance change"""
        if self.previous_distance is None:
            return 0.0
            
        distance_change = self.previous_distance - current_distance
        multiplier = 0.7 if distance_change > 0 else 1.2
        return distance_change * multiplier
    
    def _speed_reward(self, last_positions):
        """Reward for moving at appropriate speed"""
        if len(last_positions) < 2:
            return 0.0
            
        current_speed = np.linalg.norm(last_positions[-1] - last_positions[-2]) / 0.1
        speed_factor = 0.5 if (self.previous_distance is None or 
                              (self.previous_distance - np.linalg.norm(last_positions[-1] - last_positions[-2])) > 0) else 0.2
        return np.sqrt(current_speed) * speed_factor
    
    def _alignment_penalty(self, current_position, goal_position, orientation):
        """Penalty for being misaligned with goal"""
        angle_to_goal = math.atan2(goal_position[1] - current_position[1],
                                 goal_position[0] - current_position[0])
        angle_error = (angle_to_goal - orientation[0]) % (2 * math.pi)
        if angle_error > math.pi:
            angle_error = angle_error - 2 * math.pi
        return -0.2 * math.tanh(2 * angle_error)
    
    def _obstacle_penalty(self, lidar_data):
        """Penalty for being close to obstacles"""
        min_lidar = np.min(lidar_data)
        return -10.0 * max(0, 0.5 - min_lidar) if min_lidar < 0.5 else 0.0
    
    def _success_bonus(self, distance_to_goal):
        """Large bonus for reaching goal"""
        return 100.0 if distance_to_goal < self.success_threshold else 0.0
    
    def _time_penalty(self):
        """Small time penalty"""
        return -0.002


class RecoverySystem:
    """Handles all recovery behaviors"""
    
    def __init__(self):
        self.recovery_duration = 0
        self.recovery_cooldown = 0
        self.collision_recovery_steps = 0
        self.last_recovery_type = "none"
        self.safe_direction = None
        self.safe_direction_steps = 0
        self.turn_persistence = 0
        self.last_turn_direction = 0
        
        # Parameters
        self.off_course_threshold = math.pi / 16
        self.collision_threshold = 0.3
        self.max_recovery_steps = 20
        self.min_recovery_cooldown = 10
        self.max_turn_persistence = 15
        self.collision_avoidance_duration = 30
        self.max_collision_recovery = 15
        
        self.collision_memory = []
        self.max_collision_memory = 5
        self.blocked_directions = []
    
    def update(self, action, current_position, goal_position, orientation, lidar_data, 
               last_positions, oscillation_counter, logger):
        """Update recovery system and return modified action if needed"""
        recovery_active = False
        recovery_action = np.zeros(4)
        recovery_type = "none"
        
        # Update cooldown
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
        
        # Handle collision recovery
        collision_info = self._check_collision(lidar_data)
        if collision_info:
            recovery_active, recovery_action, recovery_type = self._handle_collision_recovery(
                collision_info, orientation, recovery_active, recovery_action, recovery_type)
        
        # Handle safe direction avoidance
        elif self.safe_direction_steps > 0:
            action = self._apply_safe_direction_bias(action, orientation)
            self.safe_direction_steps -= 1
        
        # Handle course correction
        elif (not recovery_active and self.recovery_cooldown <= 0 and
              self._needs_course_correction(current_position, goal_position, orientation)):
            recovery_active, recovery_action, recovery_type = self._handle_course_correction(
                current_position, goal_position, orientation, recovery_active, 
                recovery_action, recovery_type)
        
        # Handle stuck recovery
        elif (not recovery_active and self.recovery_cooldown <= 0 and
              self._is_stuck(last_positions, current_position, goal_position)):
            recovery_active, recovery_action, recovery_type = self._handle_stuck_recovery(
                recovery_active, recovery_action, recovery_type)
        
        # Apply recovery if active
        if recovery_active:
            action = self._blend_actions(action, recovery_action, recovery_type)
            oscillation_counter = 0  # Reset oscillation counter during recovery
            
            if recovery_type != self.last_recovery_type:
                logger.info(f"Recovery: {recovery_type}", throttle_duration_sec=0.5)
            self.last_recovery_type = recovery_type
        
        return action, recovery_active, oscillation_counter
    
    def _check_collision(self, lidar_data):
        """Check for imminent collisions"""
        front_zone = np.concatenate([lidar_data[:30], lidar_data[-30:]])
        front_min = np.min(front_zone)
        
        if front_min < self.collision_threshold:
            min_idx = np.argmin(lidar_data)
            collision_angle = math.radians(min_idx)
            
            # Add to blocked directions
            if len(self.blocked_directions) > 5:
                self.blocked_directions.pop(0)
            self.blocked_directions.append(collision_angle)
            
            return {'distance': front_min, 'angle': collision_angle}
        return None
    
    def _handle_collision_recovery(self, collision_info, orientation, recovery_active, recovery_action, recovery_type):
        """Handle collision recovery sequence"""
        # Store collision in memory
        if len(self.collision_memory) >= self.max_collision_memory:
            self.collision_memory.pop(0)
        self.collision_memory.append(collision_info)
        
        # Determine safe direction
        self.safe_direction = (collision_info['angle'] + math.pi) % (2 * math.pi)
        self.safe_direction_steps = self.collision_avoidance_duration
        
        # Staged recovery sequence
        recovery_type = "emergency"
        recovery_active = True
        self.collision_recovery_steps += 1
        
        if self.collision_recovery_steps < 10:  # Back up
            recovery_action = np.array([-0.8, -0.8, -0.8, -0.8])
        elif self.collision_recovery_steps < 25:  # Turn to safe direction
            angle_to_safe = (self.safe_direction - orientation[0]) % (2 * math.pi)
            turn_dir = 1 if angle_to_safe < math.pi else -1
            recovery_action = np.array([-0.6 * turn_dir, 0.6 * turn_dir, -0.6 * turn_dir, 0.6 * turn_dir])
        else:  # Move in safe direction
            recovery_action = np.array([0.5, 0.5, 0.5, 0.5])
        
        if self.collision_recovery_steps > 40:
            self.collision_recovery_steps = 0
            self.recovery_cooldown = 20
        
        return recovery_active, recovery_action, recovery_type
    
    def _apply_safe_direction_bias(self, action, orientation):
        """Bias actions toward safe direction"""
        angle_to_safe = (self.safe_direction - orientation[0]) % (2 * math.pi)
        safe_factor = max(0, 1 - (angle_to_safe / math.pi))
        
        # Reduce dangerous actions
        action = action * (0.3 + 0.7 * safe_factor)
        
        # Add bias toward safe direction
        turn_bias = 0.2 * (1 if angle_to_safe < math.pi else -1)
        action += np.array([-turn_bias, turn_bias, -turn_bias, turn_bias])
        
        return action
    
    def _needs_course_correction(self, current_position, goal_position, orientation):
        """Check if robot needs course correction"""
        angle_to_goal = math.atan2(goal_position[1] - current_position[1],
                                 goal_position[0] - current_position[0])
        angle_diff = abs(orientation[0] - angle_to_goal)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        return angle_diff > self.off_course_threshold
    
    def _handle_course_correction(self, current_position, goal_position, orientation, 
                                recovery_active, recovery_action, recovery_type):
        """Handle course correction recovery"""
        self.recovery_duration += 1
        angle_to_goal = math.atan2(goal_position[1] - current_position[1],
                                 goal_position[0] - current_position[0])
        turn_dir = 1 if (angle_to_goal - orientation[0]) % (2 * math.pi) < math.pi else -1
        
        # Maintain turn direction
        if self.last_recovery_type != "course_correction":
            self.turn_persistence = 0
            self.last_turn_direction = turn_dir
        
        if self.turn_persistence < self.max_turn_persistence:
            turn_dir = self.last_turn_direction
            self.turn_persistence += 1
        
        # Create correction action
        angle_diff = abs(orientation[0] - angle_to_goal)
        rotational_component = 0.6 * turn_dir
        forward_component = 0.4 if angle_diff < math.pi / 2 else 0.1
        
        recovery_action = np.array([
            -rotational_component + forward_component,
            rotational_component + forward_component,
            -rotational_component + forward_component,
            rotational_component + forward_component
        ])
        
        recovery_active = True
        recovery_type = "course_correction"
        
        # Exit condition
        if angle_diff < math.pi / 6:
            self.recovery_duration = 0
            self.recovery_cooldown = 10
        
        return recovery_active, recovery_action, recovery_type
    
    def _is_stuck(self, last_positions, current_position, goal_position):
        """Check if robot is stuck"""
        if len(last_positions) < 15:
            return False
            
        recent_movement = np.linalg.norm(last_positions[-1] - last_positions[0])
        distance_to_goal = np.linalg.norm(current_position - goal_position)
        
        return (recent_movement < 0.03 and distance_to_goal > 0.5)
    
    def _handle_stuck_recovery(self, recovery_active, recovery_action, recovery_type):
        """Handle stuck recovery"""
        self.recovery_duration += 1
        pattern = self.recovery_duration % 6
        
        if pattern == 0:
            recovery_action = np.array([0.5, -0.5, 0.5, -0.5])  # Right strafe
        elif pattern == 1:
            recovery_action = np.array([-0.5, 0.5, -0.5, 0.5])  # Left strafe
        elif pattern == 2:
            recovery_action = np.array([0.7, 0.7, 0.7, 0.7])   # Forward
        elif pattern == 3:
            recovery_action = np.array([-0.4, -0.4, -0.4, -0.4])  # Backward
        else:
            recovery_action = np.random.uniform(-0.3, 0.3, 4)  # Random wiggle
        
        recovery_active = True
        recovery_type = "stuck_recovery"
        
        if self.recovery_duration > 50:
            self.recovery_duration = 0
            self.recovery_cooldown = 20
        
        return recovery_active, recovery_action, recovery_type
    
    def _blend_actions(self, original_action, recovery_action, recovery_type):
        """Blend original and recovery actions"""
        if recovery_type == "emergency":
            blend_factor = 1.0  # Full override
        else:
            blend_factor = min(1.0, 0.7 + 0.3 * (self.recovery_duration / 10))
        
        return original_action * (1 - blend_factor) + recovery_action * blend_factor
    
    def reset(self):
        """Reset recovery system state"""
        self.recovery_duration = 0
        self.recovery_cooldown = 0
        self.collision_recovery_steps = 0
        self.last_recovery_type = "none"
        self.turn_persistence = 0
        self.safe_direction_steps = 0


class SpeedController:
    """Handles speed control and boosting"""
    
    def __init__(self):
        self.normal_speed = 20.0
        self.boosted_speed = 30.0
        self.min_speed = 5.0
        self.distance_threshold = 1.0
        self.min_position_change = 0.03
        
        self.boost_active = False
        self.boost_counter = 0
        self.boost_duration = 20
    
    def get_max_speed(self, progress, lidar_data, recovery_active, distance_to_goal, last_positions):
        """Calculate appropriate maximum speed"""
        # Handle speed boost
        if self.boost_active:
            self.boost_counter += 1
            if self.boost_counter > self.boost_duration:
                self.boost_active = False
        elif (progress < self.min_position_change and 
              len(last_positions) == 15 and 
              np.min(lidar_data) > 0.8 and 
              not recovery_active):
            self.boost_active = True
            self.boost_counter = 0
        
        base_speed = self.boosted_speed if self.boost_active else self.normal_speed
        
        # Reduce speed when close to goal
        if distance_to_goal < self.distance_threshold:
            t = min(1.0, distance_to_goal / self.distance_threshold)
            speed_scale = self.min_speed + (base_speed - self.min_speed) * (1 - (1 - t) ** 2)
            return np.clip(speed_scale, self.min_speed, base_speed)
        
        return base_speed


class MecanumRLController(Node, gym.Env):
    """Main RL controller for Mecanum robot"""
    
    def __init__(self):
        super().__init__('mecanum_rl_controller')
        gym.Env.__init__(self)
        
        # Initialize components
        self.reward_calculator = RewardCalculator()
        self.recovery_system = RecoverySystem()
        self.speed_controller = SpeedController()
        
        # ROS setup
        self.wheel_vel_pub = self.create_publisher(Float64MultiArray, '/wheel_velocity_controller/commands', 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.goal_pos_sub = self.create_subscription(GoalCurrentPose, '/goal_current_pose', self.goal_cpos_callback, 10)
        
        # State variables
        self.current_position = np.zeros(2)
        self.current_orientation = np.zeros(1)
        self.goal_position = None
        self.lidar_data = np.zeros(360)
        
        # Tracking buffers
        self.last_positions = []
        self.last_actions = []
        
        # RL parameters
        self.episode_length = 1000
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.oscillation_counter = 0
        self.max_oscillations = 5
        
        # Action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(low=0, high=30, shape=(360,), dtype=np.float32),
            "current_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "orientation": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "goal_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "goal_distance": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        # Initialize model and timer
        self.model = self._init_model()
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Mecanum RL Controller initialized")

    def _init_model(self):
        """Initialize the PPO model"""
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            ortho_init=True,
            log_std_init=-1.0
        )
        
        return PPO(
            "MultiInputPolicy",
            self,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.15,
            max_grad_norm=0.5,
            ent_coef=0.02
        )

    def goal_cpos_callback(self, msg):
        """Handle goal and current pose updates"""
        new_goal_x = msg.goal.pose.position.x
        new_goal_y = msg.goal.pose.position.y
        new_goal = np.array([new_goal_x, new_goal_y])
        
        # Update goal if significantly changed
        if self.goal_position is None or np.linalg.norm(new_goal - self.goal_position) > 0.5:
            self.goal_position = new_goal
            self.get_logger().info(f"New goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})",
                                 throttle_duration_sec=1.0)
            self.reset()
        else:
            self.goal_position = new_goal

        # Update current pose
        self.current_position[0] = msg.current_transform.transform.translation.x
        self.current_position[1] = msg.current_transform.transform.translation.y
        
        # Update orientation
        q = msg.current_transform.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_orientation[0] = math.atan2(siny_cosp, cosy_cosp)
    
    def lidar_callback(self, msg):
        """Process lidar data"""
        ranges = np.array(msg.ranges)
        if len(ranges) > 360:
            step = len(ranges) // 360
            self.lidar_data = ranges[::step][:360]
        else:
            self.lidar_data = ranges
        
        self.lidar_data = np.nan_to_num(self.lidar_data, posinf=30.0, neginf=0.0)
    
    def get_obs(self):
        """Get current observation"""
        if self.goal_position is None:
            self.get_logger().warn("No goal position set! Using zero position")
            self.goal_position = np.zeros(2)
        
        return {
            "lidar": self.lidar_data,
            "current_position": self.current_position,
            "orientation": self.current_orientation,
            "goal_position": self.goal_position,
            "goal_distance": np.array([np.linalg.norm(self.current_position - self.goal_position)])
        }
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment"""
        # Update tracking data
        self._update_tracking_data(action)
        
        # Handle recovery behaviors
        action, recovery_active, self.oscillation_counter = self.recovery_system.update(
            action, self.current_position, self.goal_position, self.current_orientation,
            self.lidar_data, self.last_positions, self.oscillation_counter, self.get_logger()
        )
        
        # Calculate speed and apply action
        progress = self._calculate_progress()
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        max_speed = self.speed_controller.get_max_speed(
            progress, self.lidar_data, recovery_active, distance_to_goal, self.last_positions
        )
        
        real_action = np.tanh(action) * max_speed
        self._publish_action(real_action)
        
        # Calculate reward and update state
        reward, reward_components = self.reward_calculator.compute(
            self.current_position, self.goal_position, self.current_orientation,
            self.lidar_data, self.last_positions
        )
        
        self.episode_reward += reward
        self.current_step += 1
        self.done = self._check_termination(recovery_active)
        
        # Log reward components
        self.get_logger().info(
            f"Reward: Dir={reward_components['movement_direction']:.2f} "
            f"Dist={reward_components['distance_reduction']:.2f} "
            f"Speed={reward_components['speed']:.2f}",
            throttle_duration_sec=1.0
        )
        
        return self.get_obs(), reward, self.done, {
            "distance_to_goal": distance_to_goal,
            "episode_reward": self.episode_reward,
            "recovery_active": recovery_active
        }
    
    def _update_tracking_data(self, action):
        """Update position and action tracking buffers"""
        self.last_positions.append(self.current_position.copy())
        if len(self.last_positions) > 15:
            self.last_positions.pop(0)
        
        self.last_actions.append(action.copy())
        if len(self.last_actions) > 10:
            self.last_actions.pop(0)
        
        # Detect oscillation
        if len(self.last_actions) == 10:
            action_changes = np.sum(np.abs(np.diff(self.last_actions, axis=0)))
            if action_changes > 5.0:
                self.oscillation_counter += 1
            else:
                self.oscillation_counter = max(0, self.oscillation_counter - 1)
    
    def _calculate_progress(self):
        """Calculate recent movement progress"""
        if len(self.last_positions) == 15:
            return np.linalg.norm(self.last_positions[-1] - self.last_positions[0])
        return 0.0
    
    def _publish_action(self, action):
        """Publish wheel velocities"""
        vel_msg = Float64MultiArray()
        vel_msg.data = action.tolist()
        self.wheel_vel_pub.publish(vel_msg)
    
    def _check_termination(self, recovery_active):
        """Check if episode should terminate"""
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        
        return (
            self.current_step >= self.episode_length or
            (self.reward_calculator.previous_distance is not None and 
             distance_to_goal > self.reward_calculator.previous_distance * 2.0 and 
             self.current_step > 30) or
            (recovery_active and self.recovery_system.recovery_duration > 100) or
            self.oscillation_counter > self.max_oscillations * 2
        )
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.last_positions = []
        self.last_actions = []
        self.oscillation_counter = 0
        
        self.reward_calculator.previous_distance = None
        self.recovery_system.reset()
        
        return self.get_obs(), {}
    
    def control_loop(self):
        """Main control loop"""
        if self.goal_position is None:
            self.get_logger().warn("Waiting for first goal position...", throttle_duration_sec=1.0)
            return
    
        if not self.done:
            obs = self.get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            self.step(action)
        else:
            self.get_logger().info(
                f"Episode ended. Reward: {self.episode_reward:.2f}, "
                f"Distance: {np.linalg.norm(self.current_position - self.goal_position):.2f}m"
            )
            self.reset()

def main(args=None):
    rclpy.init(args=args)
    node = MecanumRLController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()