# Simulation of a Four-Wheeled Mecanum Robot in Gazebo Harmonic (ROS 2 Jazzy)

This repository provides a simulation of a four-wheeled Mecanum robot using **Gazebo Harmonic** and **ROS 2 Jazzy**.

## Navigation Methods
The simulation supports two navigation approaches:

1. **Nav2 (Navigation Stack 2)** – Currently not optimized.
2. **Reinforcement Learning (RL)** – Implementation is in progress.

## 📦 Repository

**GitHub Link**: [https://github.com/EbyGunner/mecanum_robot.git](https://github.com/EbyGunner/mecanum_robot.git)

---

## 🛠️ Requirements

- ROS 2 Jazzy
- Nav2
- Python3 packages: `gymnasium` and `stable_baselines3`

---

## ✅ Installation Steps

### 1. Setup a ROS 2 Humble Workspace

```bash
mkdir -p ~/mecanum_robot/src
cd ~/mecanum_robot/src
```
---

### 2. Clone the Repository

⚠️ DO NOT create a sub-folder; clone the repository directly inside `src`.

```bash
git clone https://github.com/EbyGunner/mecanum_robot.git
```
---
### 3. Install Required Python Packages
Local installation of mediapipe in project directory:

To avoid installing mediapipe globally:
```bash
cd ~/mecanum_robot/src/rl_control/rl_control
mkdir -p external_libraries
pip3 install stable_baselines3 --target=external_libraries
pip3 install gymnasium --target=external_libraries
```
This will install the required python libraries in the `rl_control/external_libraries` folder. The reinforcement learning script is already configured to import it from this local path.

---
### 5. Build the Workspace
```bash
cd ~/mecanum_robot
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```
---
### 6. Source the Workspace
```bash
source ~/mecanum_robot/install/setup.bash
```
---

Note: By default, the simulation runs using RL mode unless specified otherwise.
