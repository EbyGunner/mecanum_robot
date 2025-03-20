# Simulation of a Four-Wheeled Mecanum Robot in Gazebo Harmonic (ROS 2 Jazzy)

This repository provides a simulation of a four-wheeled Mecanum robot using **Gazebo Harmonic** and **ROS 2 Jazzy**.

## Navigation Methods
The simulation supports two navigation approaches:

1. **Nav2 (Navigation Stack 2)** – Currently not optimized.
2. **Reinforcement Learning (RL)** – Implementation is in progress.

## Setup Instructions

### Clone and Build the Repository
```bash
cd ~/ros2_ws/src  # Navigate to your ROS 2 workspace
git clone https://github.com/EbyGunner/mecanum_robot.git  # Clone the repository
cd ..
colcon build  # Build the workspace
source install/setup.bash  # Source the workspace
```

### Running the Simulation

**Run the RL Control Algorithm**

``` bash
ros2 launch mecanum_robot_simulation robot_main.launch.py use_RL:=true
```

**Run the Nav2 Control Algorithm**

``` bash
ros2 launch mecanum_robot_simulation robot_main.launch.py use_RL:=false
```

Note: By default, the simulation runs using Nav2 unless specified otherwise.
