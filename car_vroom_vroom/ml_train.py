#!/usr/bin/env python3
"""
Training script for the F1Tenth simulation using PPO

Script trains a model-free RL agent to driving an F1Tenth car around a race track.

Takes in LiDAR sensor data and outputs steering and speed.

Logs reward components to a csv file, and periodically saves intermediate models, as well as the best model so far.
"""
import os
import sys
import argparse
import torch
import numpy as np
import csv
from typing import Dict, Any

# Add parent directory to path for imports when running as script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from car_vroom_vroom.env_wrapper import F1TenthEnvWrapper


class RewardComponentLogger(BaseCallback):
    """
    Callback to log reward components to CSV files for analysis

    Logs cumulative reward components for each completed episode. Mostly used for debug/understanding which components are 
    dominate by the end of training
    """
    def __init__(self, csv_path: str, verbose: int = 0):
        """
        Initializes the loggers

        Args:
            csv_path: path to save the .csv file
            verbose: to print debug statements or not
        """
        super().__init__(verbose)
        self.csv_path = csv_path
        self.logged_episodes = set()
        
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        
        file_exists = os.path.exists(csv_path)
        self.csv_file = open(csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            header = ['timestep', 'total_reward', 'progress', 'speed', 'safety', 'center', 'turning', 'smooth', 'crash']
            self.csv_writer.writerow(header)
            self.csv_file.flush()
    
    def _on_step(self) -> bool:
        """
        Called at each training step. Logs episode reward components when episodes end
        """
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        # check each env to see if it completed
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and info and 'episode_reward' in info:
                episode_id = (self.num_timesteps, i)
                
                # logs the episode
                if episode_id not in self.logged_episodes:
                    self.logged_episodes.add(episode_id)
                    
                    episode_reward = info['episode_reward']
                    episode_components = info.get('episode_reward_components', {})
                    
                    #get the reward components
                    if isinstance(episode_components, dict) and len(episode_components) > 0:
                        total = episode_components.get('total', episode_reward)
                        progress = episode_components.get('progress', 0.0)
                        speed = episode_components.get('speed', 0.0)
                        safety = episode_components.get('safety', 0.0)
                        center = episode_components.get('center', 0.0)
                        turning = episode_components.get('turning', 0.0)
                        smooth = episode_components.get('smooth', 0.0)
                        crash = episode_components.get('crash', 0.0)
                    else:
                        total = episode_reward
                        progress = speed = safety = center = turning = smooth = crash = 0.0
                    
                    row = [
                        self.num_timesteps,
                        total,
                        progress,
                        speed,
                        safety,
                        center,
                        turning,
                        smooth,
                        crash
                    ]
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()
                    
                    if len(self.logged_episodes) > 1000:
                        self.logged_episodes.clear()
        
        return True
    
    def _on_training_end(self) -> None:
        if self.csv_file:
            self.csv_file.close()


def make_env(map_path: str, rank: int = 0, seed: int = 0, waypoints_path: str = None, max_speed: float = 5.0, min_speed: float = 0.0, track_width: float = 3.0):

    """
    Function to create the env for parallel traing. Each env gets a unique seed to ensure randomness and diversity.
    """
    def _init():
        env = F1TenthEnvWrapper(map_path=map_path, map_ext='.png', num_agents=1, waypoints_path=waypoints_path, max_speed=max_speed, min_speed=min_speed, track_width=track_width)
        env.seed(seed + rank)
        return env
    return _init


def main():

    """
    Main training function, sets up parallel environments, makes the PPO model, and runs training with eval, checkpoints and reward logging.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-path', type=str, 
                       default='/home/contranickted/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map')
    parser.add_argument('--total-timesteps', type=int, default=1000000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--save-freq', type=int, default=100000)
    parser.add_argument('--waypoints-path', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--max-speed', type=float, default=5.0)
    parser.add_argument('--min-speed', type=float, default=0.0)
    parser.add_argument('--track-width', type=float, default=3.0)
    
    args = parser.parse_args()
    
    # get file paths
    cwd = os.getcwd()
    args.log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.abspath(os.path.join(cwd, args.log_dir))
    args.model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.abspath(os.path.join(cwd, args.model_dir))
    
    # check for gpu
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = args.device
    

    # output locations
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"TensorBoard logs: {args.log_dir}")
    print(f"Models will be saved to: {args.model_dir}")
    

    # detect centerlines
    waypoints_path = args.waypoints_path
    if waypoints_path is None:
        map_dir = os.path.dirname(args.map_path)
        map_name = os.path.basename(args.map_path)
        base_name = map_name.replace('_map', '')
        centerline_file = os.path.join(map_dir, f"{base_name}_centerline.csv")
        if os.path.exists(centerline_file):
            waypoints_path = centerline_file
            print(f"Auto-detected centerline file: {waypoints_path}")
        else:
            print(f"No centerline file found at {centerline_file}, centering will use lidar only")
    

    # runs envs in parallel
    print(f"Creating {args.n_envs} parallel environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(args.map_path, i, waypoints_path=waypoints_path, max_speed=args.max_speed, min_speed=args.min_speed, track_width=args.track_width) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(args.map_path, 0, waypoints_path=waypoints_path, max_speed=args.max_speed, min_speed=args.min_speed, track_width=args.track_width)])
    

    # make eval environments
    def make_monitored_env():
        from stable_baselines3.common.monitor import Monitor
        env = F1TenthEnvWrapper(map_path=args.map_path, map_ext='.png', num_agents=1, waypoints_path=waypoints_path, max_speed=args.max_speed, min_speed=args.min_speed, track_width=args.track_width)
        env = Monitor(env, os.path.join(args.log_dir, 'eval'))
        return env
    
    eval_env = DummyVecEnv([make_monitored_env])
    
    # 2 hidden layers, 2048 nodes each
    policy_kwargs = dict(
        net_arch=[2048, 2048],
        activation_fn=torch.nn.ReLU,
    )
    
    # resume or no
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
        model = PPO.load(args.resume_from, device=device)
        model.set_env(env)
    else:
        # some ppo policy values that I found from https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
        # mostly via trial and error, and looking up good values to use
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
        )
    
    # Periodically evaluates the models saves best version
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.save_freq,
        deterministic=True,
        render=False
    )
    
    # Checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.model_dir,
        name_prefix='ppo_f1tenth',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    

    # logs reward to csv
    csv_log_path = os.path.join(args.log_dir, 'reward_components.csv')
    reward_logger = RewardComponentLogger(csv_path=csv_log_path)
    
    # Start training
    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Model will be saved to {args.model_dir}")
    print(f"Logs will be saved to {args.log_dir}")
    print(f"Reward components will be logged to: {csv_log_path}")
    
    try:
        # start running training
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback, reward_logger],
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    #save final model
    final_model_path = os.path.join(args.model_dir, 'ppo_f1tenth_final')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    

    # cleanup
    env.close()
    eval_env.close()
    
    print("Training completed!")


if __name__ == '__main__':
    main()

