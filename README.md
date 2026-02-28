# Model-Free Reinforcement Learning Agent for F1Tenth Autonomous Driving

This is a reinforcement learning project for training an autonomous F1Tenth car. The agent learns to navigate race tracks using LiDAR sensor data to output steering and speed commands.

## Overview

This project implements a model-free reinforcement learning solution for F1Tenth. The agent uses PPO (Proximal Policy Optimization) from Stable-Baselines3 to learn optimal driving policies through trial and error in simulation. The trained model can be deployed for real-time inference in ROS 2 environments.

### Key Features

- **PPO-based Training**: Uses Stable-Baselines3's PPO algorithm with customizable hyperparameters
- **Reward Shaping**: Weighted multi-component reward function including:
  - Progress tracking (forward movement along track)
  - Speed incentives
  - Safety penalties (wall distance)
  - Centering rewards (track center alignment)
  - Smoothness penalties (reducing erratic steering)
  - Crash penalities
- **Parallel Training**: Supports multiple parallel environments for faster training
- **Waypoint Support**: Optional centerline waypoint tracking for accurate progress measurement and centering reward
- **ROS 2 Integration**: Inference node for real-time deployment
- **Comprehensive Logging**: TensorBoard logs, reward component breakdowns, and model checkpoints

## Requirements

### System Requirements
- Ubuntu 20.04.6
- ROS 2 Foxy
- Python 3.8
- CUDA-capable GPU (optional, but speeds up training)

### Dependencies

#### ROS 2 Packages
- `rclpy`
- `sensor_msgs`
- `nav_msgs`
- `ackermann_msgs`
- `geometry_msgs`
- `std_msgs`
- `launch`
- `launch_ros`

#### Python Packages
See `requirements.txt` for details:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `torchaudio>=2.0.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `tensorboard>=2.10.0`
- `stable-baselines3`
- `gym`
- `f110_gym` (F1Tenth gym environment)

## Installation

### 1. Clone and Build

```bash
source /opt/ros/foxy/setup.bash
cd ~/sim_ws/src
git clone https://github.com/CPEN-391/Project_GA6.git
cd ~/sim_ws
colcon build --packages-select car_vroom_vroom
source install/setup.bash
```

### 2. Install Python Dependencies

```bash
cd ~/sim_ws/src/car_vroom_vroom
pip install -r requirements.txt
pip install stable-baselines3[extra]
pip install gym f110-gym
pip install shimmy
```

### 3. Install F1Tenth Gym Environment

Follow the official F1Tenth Gym installation instructions

## Project Structure

```
car_vroom_vroom/
├── car_vroom_vroom/          # Main Python package
│   ├── env_wrapper.py        # Gym-compatible F1Tenth environment wrapper
│   ├── ml_train.py           # Training script with PPO
│   └── ml_inference.py       # ROS 2 inference node
│   └── safety_node.py        # AEB safety node
├── config/                   # Configuration files
│   ├── training.yaml         # Training configuration
│   └── inference.yaml        # Inference configuration
├── launch/                   # ROS 2 launch files
│   ├── training_launch.py    # Launch training script
│   └── inference_launch.py   # Launch inference node
├── models/                   # Trained model checkpoints (generated)
├── logs/                     # Training logs and TensorBoard data (generated)
├── requirements.txt          # Python dependencies
├── setup.py                  # ROS 2 package setup
└── package.xml               # ROS 2 package metadata
```

## Usage

### Training

Train a new model using the training script:

```bash
# Basic training
ros2 run car_vroom_vroom ml_train \
    --map-path /path/to/maps/Spielberg_map \
    --total-timesteps 1000000 \
    --n-envs 8

# With all options
ros2 run car_vroom_vroom ml_train \
    --map-path /path/to/maps/Spielberg_map \
    --total-timesteps 10000000 \
    --learning-rate 3e-4 \
    --n-envs 8 \
    --n-steps 4096 \
    --batch-size 256 \
    --n-epochs 10 \
    --device cuda \
    --log-dir ./logs \
    --model-dir ./models \
    --save-freq 100000 \
    --waypoints-path /path/to/centerline.csv \
    --max-speed 5.0 \
    --min-speed 0.0 \
    --track-width 3.0

# Resume from checkpoint
ros2 run car_vroom_vroom ml_train \
    --resume-from ./models/ppo_f1tenth_4800000_steps.zip \
    --total-timesteps 10000000
```

#### Training Arguments

- `--map-path`: Path to F1Tenth map (without extension)
- `--total-timesteps`: Total training timesteps (default: 1,000,000)
- `--learning-rate`: PPO learning rate (default: 3e-4)
- `--n-envs`: Number of parallel environments (default: 8)
- `--n-steps`: Steps per environment per update (default: 4096)
- `--batch-size`: Minibatch size (default: 256)
- `--n-epochs`: Number of optimization epochs per update (default: 10)
- `--device`: Device for training ('auto', 'cuda', or 'cpu')
- `--log-dir`: Directory for logs (default: ./logs)
- `--model-dir`: Directory for model checkpoints (default: ./models)
- `--save-freq`: Frequency of model saves in timesteps (default: 100000)
- `--waypoints-path`: Path to centerline waypoints CSV (optional)
- `--max-speed`: Maximum speed in m/s (default: 5.0)
- `--min-speed`: Minimum speed in m/s (default: 0.0)
- `--track-width`: Estimated track width in meters (default: 3.0)

#### Monitoring Training


Reward component breakdowns are logged to `logs/reward_components.csv`.

### Inference

Run inference on a trained model in a ROS 2 environment:

```bash
# Using launch file
ros2 launch car_vroom_vroom inference_launch.py \
    model_path:=./models/ppo_f1tenth_final.zip

# Direct node execution
ros2 run car_vroom_vroom ml_inference \
    --ros-args \
    -p model_path:=./models/ppo_f1tenth_final.zip \
    -p scan_topic:=/ego_racecar/scan \
    -p odom_topic:=/ego_racecar/odom \
    -p drive_topic:=/ego_racecar/drive \
    -p max_range:=30.0 \
    -p device:=auto \
    -p min_speed:=0.5 \
    -p max_speed:=5.0
```

#### Inference Parameters

- `model_path`: Path to trained model ZIP file (required)
- `scan_topic`: LiDAR scan topic (default: `/scan`)
- `odom_topic`: Odometry topic (default: `/ego_racecar/odom`)
- `drive_topic`: Control command topic (default: `/drive`)
- `max_range`: Maximum LiDAR range in meters (default: 30.0)
- `device`: Inference device ('auto', 'cuda', or 'cpu')
- `min_speed`: Minimum speed in m/s (default: 0.5)
- `max_speed`: Maximum speed in m/s (default: 5.0)

### Real-World Deployment on Physical Track

The inference node is designed to work with real F1Tenth hardware. The same inference code runs in both simulation and on physical cars.

#### Prerequisites for Real Hardware

1. **ROS 2 Setup on F1Tenth Car**: Ensure ROS 2 is properly installed and configured on the car's computer
2. **Sensor Drivers Running**: LiDAR and odometry sensors must be publishing to ROS topics
3. **Safety**: Have an emergency stop mechanism ready before running on real hardware

#### Required ROS Topics

The inference node expects the following ROS 2 topics to be available:

- **LiDAR Scan**: `/ego_racecar/scan` (or configured via `scan_topic` parameter)
  - Message type: `sensor_msgs/LaserScan`
  - Must publish LiDAR scan data with ranges in meters
- **Odometry**: `/ego_racecar/odom` (or configured via `odom_topic` parameter)
  - Message type: `nav_msgs/Odometry`
  - Current odometry data (used for reference, but model primarily uses LiDAR)
- **Drive Commands**: `/ego_racecar/drive` (or configured via `drive_topic` parameter)
  - Message type: `ackermann_msgs/AckermannDriveStamped`
  - This is where the inference node **publishes** control commands


## Environment Wrapper

The `F1TenthEnvWrapper` provides a Gym-compatible interface to the F1Tenth simulator with:

- **Observation Space**: Normalized LiDAR scan (1080 beams, [0, 1] range)
- **Action Space**: `[steering_angle (radians), speed (m/s)]`
  - Steering: -0.4189 to 0.4189 radians (±24 degrees)
  - Speed: min_speed to max_speed m/s

### Reward Components

The reward function combines multiple components:

1. **Progress** (weight: 3.0): Rewards forward movement along track
2. **Speed** (weight: 0.5): Encourages maintaining forward velocity
3. **Safety** (weight: 2.0): Penalizes proximity to walls
4. **Centering** (weight: 0.5): Keeps car aligned with track center
5. **Turning** (weight: 0.5): Small penalty for turning (allows necessary turns)
6. **Smoothness** (weight: 0.5): Penalizes erratic steering and oscillations
7. **Crash Penalty**: Large penalty (-35,000) for collisions

Episodes terminate on crash or time limit (default: 300 seconds).

## Configuration Files

### Training Configuration (`config/training.yaml`)

```yaml
training:
  map_path: '/path/to/maps/levine'
  total_timesteps: 1000000
  learning_rate: 0.0003
  n_envs: 4
  device: 'auto'
  log_dir: './logs'
  model_dir: './models'
  save_freq: 50000
```

### Inference Configuration (`config/inference.yaml`)

```yaml
inference:
  ros__parameters:
    model_path: './models/ppo_f1tenth_final.zip'
    scan_topic: '/ego_racecar/scan'
    odom_topic: '/ego_racecar/odom'
    drive_topic: '/ego_racecar/drive'
    max_range: 30.0
    device: 'auto'
```

## Model Architecture

- **Policy**: MLP (Multi-Layer Perceptron)
- **Hidden Layers**: 2 layers with 2048 units each
- **Activation**: ReLU
- **Algorithm**: PPO with:
  - Gamma: 0.99
  - GAE Lambda: 0.95
  - Clip Range: 0.2
  - Entropy Coefficient: 0.01
  - Value Function Coefficient: 0.5
  - Max Gradient Norm: 0.5

## Output Files

### Training Outputs

- **Models**: Saved in `models/` directory
  - `best_model.zip`: Best performing model (based on evaluation)
  - `ppo_f1tenth_<timesteps>_steps.zip`: Periodic checkpoints
  - `ppo_f1tenth_final.zip`: Final model after training completes

- **Logs**: Saved in `logs/` directory
  - TensorBoard event files (viewable with TensorBoard)
  - `reward_components.csv`: Episode-by-episode reward breakdown
  - Evaluation monitor CSV files




## Contributors

- Andy Lu
- Kyan Galvez
- Nick Liu
- Tong Wu
- Justin Chan
- Austen Wang