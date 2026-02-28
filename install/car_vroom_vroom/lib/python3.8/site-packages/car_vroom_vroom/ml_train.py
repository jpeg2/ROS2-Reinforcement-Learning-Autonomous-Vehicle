#!/usr/bin/env python3
"""
Model-free ML training script for F1Tenth car using PPO.
Uses GPU if available.
"""
import os
import sys
import argparse
import torch
import numpy as np

# Add parent directory to path for imports when running as script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from car_vroom_vroom.env_wrapper import F1TenthEnvWrapper


def make_env(map_path: str, rank: int = 0, seed: int = 0, waypoints_path: str = None, max_speed: float = 5.0, min_speed: float = 0.0, track_width: float = 3.0):
    """
    Create a single environment.
    
    Args:
        map_path: Path to the map file
        rank: Process rank for seeding
        seed: Random seed
        waypoints_path: Optional path to centerline waypoints
        max_speed: Maximum speed in m/s
        min_speed: Minimum speed in m/s
        track_width: Track width in meters
    """
    def _init():
        env = F1TenthEnvWrapper(map_path=map_path, map_ext='.png', num_agents=1, waypoints_path=waypoints_path, max_speed=max_speed, min_speed=min_speed, track_width=track_width)
        env.seed(seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train F1Tenth car using model-free RL')
    parser.add_argument('--map-path', type=str, 
                       default='/home/contranickted/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map',
                       help='Path to map file (without extension)')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total number of training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n-envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=4096,
                       help='Number of steps per update (default: 4096)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of epochs per update (default: 10)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logging')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory for saving models')
    parser.add_argument('--save-freq', type=int, default=100000,
                       help='Frequency of model saves (in timesteps)')
    parser.add_argument('--waypoints-path', type=str, default=None,
                       help='Optional path to Nx2 centerline waypoints file for track progress reward')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to an existing .zip policy to resume training from')
    parser.add_argument('--max-speed', type=float, default=5.0,
                       help='Maximum speed in m/s (default: 5.0)')
    parser.add_argument('--min-speed', type=float, default=0.0,
                       help='Minimum speed in m/s (default: 0.0)')
    parser.add_argument('--track-width', type=float, default=3.0,
                       help='Track width in meters (default: 3.0)')
    
    args = parser.parse_args()
    
    # Resolve directories to absolute paths for predictable saving
    cwd = os.getcwd()
    args.log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.abspath(os.path.join(cwd, args.log_dir))
    args.model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.abspath(os.path.join(cwd, args.model_dir))
    
    # Check GPU availability
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = args.device
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"TensorBoard logs: {args.log_dir}")
    print(f"Models will be saved to: {args.model_dir}")
    
    # Automatically find centerline file in same directory as map if not specified
    waypoints_path = args.waypoints_path
    if waypoints_path is None:
        # Derive centerline path from map path
        # e.g., /path/to/Spielberg_map -> /path/to/Spielberg_centerline.csv
        map_dir = os.path.dirname(args.map_path)
        map_name = os.path.basename(args.map_path)
        # Remove any "_map" suffix if present
        base_name = map_name.replace('_map', '')
        centerline_file = os.path.join(map_dir, f"{base_name}_centerline.csv")
        if os.path.exists(centerline_file):
            waypoints_path = centerline_file
            print(f"Auto-detected centerline file: {waypoints_path}")
        else:
            print(f"Warning: No centerline file found at {centerline_file}, centering will use lidar only")
    
    # Create vectorized environment
    print(f"Creating {args.n_envs} parallel environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(args.map_path, i, waypoints_path=waypoints_path, max_speed=args.max_speed, min_speed=args.min_speed, track_width=args.track_width) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(args.map_path, 0, waypoints_path=waypoints_path, max_speed=args.max_speed, min_speed=args.min_speed, track_width=args.track_width)])
    
    # Create evaluation environment
    # Monitor needs the unwrapped env, so we wrap the inner env first
    def make_monitored_env():
        from stable_baselines3.common.monitor import Monitor
        env = F1TenthEnvWrapper(map_path=args.map_path, map_ext='.png', num_agents=1, waypoints_path=waypoints_path, max_speed=args.max_speed, min_speed=args.min_speed, track_width=args.track_width)
        env = Monitor(env, os.path.join(args.log_dir, 'eval'))
        return env
    
    eval_env = DummyVecEnv([make_monitored_env])
    
    # Define PPO hyperparameters optimized for continuous control
    policy_kwargs = dict(
        net_arch=[1024, 1024],  # Two hidden layers with 1024 units each (doubled from 512)
        activation_fn=torch.nn.ReLU,
    )
    
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
        # Load model without environment first to avoid action space mismatch
        # Then set the new environment
        model = PPO.load(args.resume_from, device=device)
        model.set_env(env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,  # Steps per environment before update (increased for more GPU usage)
            batch_size=args.batch_size,  # Larger batch size for more GPU computation
            n_epochs=args.n_epochs,  # Number of epochs for optimization
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE lambda parameter
            clip_range=0.2,  # PPO clip range
            ent_coef=0.01,  # Entropy coefficient
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Maximum gradient norm
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
        )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.save_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.model_dir,
        name_prefix='ppo_f1tenth',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Train the model
    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Model will be saved to {args.model_dir}")
    print(f"Logs will be saved to {args.log_dir}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=False  # Disable progress bar to avoid tqdm dependency
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'ppo_f1tenth_final')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("Training completed!")


if __name__ == '__main__':
    main()

