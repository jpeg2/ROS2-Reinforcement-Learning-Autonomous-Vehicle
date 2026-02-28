"""
Wrapper for F1Tenth Gym environment to work with RL algorithms.

This module provides a Gym-compatible wrapper around the F1Tenth racing simulator
that implements reward shaping, observation normalization, and episode management
for reinforcement learning training.

Key Features:
- Normalizes lidar observations to [0, 1] range for neural network input
- Implements multi-component reward function (progress, speed, safety, centering, smoothness)
- Supports centerline waypoint tracking for accurate progress measurement
- Handles crash detection and episode termination
- Manages episode timing and statistics
"""
import gym
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional


class F1TenthEnvWrapper(gym.Env):
    """
    Gym-compatible wrapper for F1Tenth racing environment.
    
    Wraps the F1Tenth gym environment to provide:
    - Standardized observation/action spaces for RL algorithms
    - Reward shaping with multiple components (progress, speed, safety, centering, smoothness)
    - Lidar observation normalization to [0, 1] range
    - Episode termination on crash or time limit
    - Centerline waypoint support for accurate progress tracking
    - Crash detection using environment signals and lidar distance
    
    Observation Space:
        - Normalized lidar scan: Box(shape=(1080,), low=0.0, high=1.0)
        - Represents distances to walls/obstacles in all directions
    
    Action Space:
        - Box(shape=(2,), low=[-0.4189, min_speed], high=[0.4189, max_speed])
        - [steering_angle (radians), speed (m/s)]
        - Steering range: ±24 degrees (±0.4189 radians)
    """
    
    def __init__(self, map_path: str, map_ext: str = '.png', num_agents: int = 1, waypoints_path: Optional[str] = None, max_episode_seconds: float = 300.0, sim_dt: float = 0.01, max_speed: float = 5.0, min_speed: float = 0.0, track_width: float = 3.0):
        """
        Initialize the F1Tenth environment wrapper.
        
        Args:
            map_path: Path to the map file (without extension)
            map_ext: Map image extension
            num_agents: Number of agents (1 for training)
            waypoints_path: Optional path to centerline waypoints CSV file (x_m, y_m format) for track progress reward
            max_episode_seconds: Time limit for an episode in seconds (default: 300.0 = 5 minutes)
            sim_dt: Assumed simulator step in seconds used for time accumulation
            max_speed: Maximum speed in m/s (default: 5.0)
            min_speed: Minimum speed in m/s (default: 0.0)
            track_width: Estimated track width in meters (default: 3.0) for centering reward calculation
        """
        super().__init__()
        
        self.env = gym.make('f110_gym:f110-v0',
                           map=map_path,
                           map_ext=map_ext,
                           num_agents=num_agents)
        
        # Episode timing
        self.max_episode_seconds = float(max_episode_seconds)
        self.sim_dt = float(sim_dt)
        self.step_count = 0
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.track_width = float(track_width)
        self.center_threshold = 0.2 * self.track_width  # 20% of track width from center
        
        # Action space: [steering_angle, speed]
        # Steering: -0.4189 to 0.4189 radians (-24 to 24 degrees)
        # Speed: min_speed to max_speed m/s
        self.action_space = gym.spaces.Box(
            low=np.array([-0.4189, self.min_speed]),
            high=np.array([0.4189, self.max_speed]),
            dtype=np.float32
        )
        
        # Reset environment to get observation shape
        # Use default starting position if available
        if hasattr(self.env, 'start_xs') and len(self.env.start_xs) > 0:
            init_pose = np.array([[self.env.start_xs[0], self.env.start_ys[0], self.env.start_thetas[0]]])
        else:
            init_pose = np.array([[0.0, 0.0, 0.0]])
        temp_obs, _, _, _ = self.env.reset(init_pose)
        
        # Observation space: Lidar scan (default 1080 beams)
        scan_beams = len(temp_obs['scans'][0])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(scan_beams,),
            dtype=np.float32
        )
        
        self.scan_beams = scan_beams
        self.max_range = 30.0
        self.prev_progress = 0.0
        self.prev_steer = 0.0
        self.prev_prev_steer = 0.0  # Track steering history for oscillation detection
        
        # Calculate lidar indices for ±90 degrees

        # Scan range: -2.35 to +2.35 radians, with 0 = forward
        # For ±90 degrees (±π/2 ≈ ±1.57 radians from forward):
        scan_fov = 4.7  # radians (270 degrees)
        angle_min = -scan_fov / 2.0  # -2.35 radians
        angle_inc = scan_fov / scan_beams  # angle increment per beam
        
        # Left side: +90 degrees from forward = +π/2 radians
        left_90_angle = np.pi / 2.0  # 90 degrees in radians
        left_90_idx_center = int((left_90_angle - angle_min) / angle_inc)
        # Use a small window (±5 degrees) around 90 degrees for robustness
        window_angle = 5.0 * np.pi / 180.0  # 5 degrees in radians
        window_size = int(window_angle / angle_inc)  # ~5 degrees in beam indices
        self.left_90_idx_start = np.clip(left_90_idx_center - window_size, 0, scan_beams - 1)
        self.left_90_idx_end = np.clip(left_90_idx_center + window_size, 0, scan_beams - 1)
        
        # Right side: -90 degrees from forward = -π/2 radians
        right_90_angle = -np.pi / 2.0  # -90 degrees in radians
        right_90_idx_center = int((right_90_angle - angle_min) / angle_inc)
        # Use a small window (±5 degrees) around -90 degrees for robustness
        self.right_90_idx_start = np.clip(right_90_idx_center - window_size, 0, scan_beams - 1)
        self.right_90_idx_end = np.clip(right_90_idx_center + window_size, 0, scan_beams - 1)
        
        # Waypoint-based progress tracking
        self.waypoints = None
        self.waypoint_distances = None  # Cumulative distances along waypoints
        self.use_centerline_progress = False
        if waypoints_path and os.path.exists(waypoints_path):
            self._load_waypoints(waypoints_path)
            self.use_centerline_progress = True
        
        # Episode tracking for statistics
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
    
    def _load_waypoints(self, waypoints_path: str) -> None:
        """
        Load centerline waypoints from CSV file.
        
        Expected CSV format:
        - First row: header (skipped)
        - Subsequent rows: x_m, y_m (meters)
        - Comments starting with '#' are ignored
        
        After loading, computes cumulative distances along the waypoint path
        for progress tracking.
        
        Args:
            waypoints_path: Path to CSV file containing waypoint coordinates
        
        Raises:
            Prints warning and disables waypoint tracking if loading fails
        """
        try:
            # Read CSV file, skip header if present
            data = np.loadtxt(waypoints_path, delimiter=',', skiprows=1, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Extract x, y coordinates (first two columns)
            self.waypoints = data[:, :2].astype(np.float32)
            
            # Compute cumulative distances along waypoints
            self.waypoint_distances = np.zeros(len(self.waypoints))
            for i in range(1, len(self.waypoints)):
                dist = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1])
                self.waypoint_distances[i] = self.waypoint_distances[i-1] + dist
            
            print(f"Loaded {len(self.waypoints)} waypoints from {waypoints_path}")
        except Exception as e:
            print(f"Warning: Failed to load waypoints from {waypoints_path}: {e}")
            self.waypoints = None
            self.waypoint_distances = None
            self.use_centerline_progress = False
    
    def _compute_centerline_progress(self, x: float, y: float) -> float:
        """
        Compute progress along centerline by finding the closest segment and interpolating.
        
        Projects the car's position onto the nearest centerline segment and interpolates
        the progress value. This is more stable than using just the closest waypoint,
        especially when waypoints are sparse.
        
        Algorithm:
        1. Find closest waypoint to car position
        2. Check two adjacent segments (prev->closest, closest->next)
        3. Project car onto each segment
        4. Use segment with minimum distance to car
        5. Interpolate progress along that segment
        
        Args:
            x: Car x-coordinate in meters
            y: Car y-coordinate in meters
        
        Returns:
            Cumulative distance along centerline in meters, or x-coordinate if no waypoints
        """
        if self.waypoints is None or len(self.waypoints) < 2:
            # Fallback to x-coordinate if no waypoints
            return x
        
        if self.waypoint_distances is None:
            # Fallback to x-coordinate if distances not available
            return x
        
        car_pos = np.array([x, y])
        n_waypoints = len(self.waypoints)
        
        # Find closest waypoint
        distances_to_waypoints = np.linalg.norm(self.waypoints - car_pos, axis=1)
        closest_idx = np.argmin(distances_to_waypoints)
        
        # Check neighboring segments to find which one we're closest to
        prev_idx = (closest_idx - 1) % n_waypoints
        next_idx = (closest_idx + 1) % n_waypoints
        
        # Calculate progress by projecting onto the closest segment
        best_progress = None
        min_dist_sq = float('inf')
        
        # Check the two segments adjacent to closest waypoint
        for idx1, idx2 in [(prev_idx, closest_idx), (closest_idx, next_idx)]:
            p1 = self.waypoints[idx1]
            p2 = self.waypoints[idx2]
            segment_vec = p2 - p1
            segment_len_sq = np.dot(segment_vec, segment_vec)
            
            if segment_len_sq < 1e-6:
                continue  # Skip zero-length segments
            
            # Project car position onto segment
            car_vec = car_pos - p1
            t = np.clip(np.dot(car_vec, segment_vec) / segment_len_sq, 0.0, 1.0)
            proj_point = p1 + t * segment_vec
            
            # Distance from car to projected point
            dist_sq = np.sum((car_pos - proj_point) ** 2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                # Interpolate progress along this segment
                progress1 = self.waypoint_distances[idx1] if idx1 < len(self.waypoint_distances) else 0.0
                progress2 = self.waypoint_distances[idx2] if idx2 < len(self.waypoint_distances) else progress1
                best_progress = progress1 + t * (progress2 - progress1)
        
        if best_progress is None:
            # Fallback to closest waypoint's progress
            best_progress = self.waypoint_distances[closest_idx] if closest_idx < len(self.waypoint_distances) else 0.0
        
        return float(best_progress)
    
    def _compute_distance_to_centerline(self, x: float, y: float) -> float:
        """
        Compute perpendicular distance from car to centerline.
        
        Finds the minimum perpendicular distance from the car's position to any
        centerline segment. Used for centering reward calculation.
        
        Algorithm:
        1. Find closest waypoint
        2. Check adjacent segments
        3. Project car position onto each segment
        4. Compute perpendicular distance
        5. Return minimum distance
        
        Args:
            x: Car x-coordinate in meters
            y: Car y-coordinate in meters
        
        Returns:
            Perpendicular distance in meters (0 = on centerline, positive = off centerline).
            Returns 0.0 if no waypoints available.
        """
        if self.waypoints is None or len(self.waypoints) < 2:
            return 0.0
        
        car_pos = np.array([x, y])
        
        # Find the two closest waypoints to form a line segment
        distances = np.linalg.norm(self.waypoints - car_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Check neighboring waypoints to find the segment we're closest to
        n_waypoints = len(self.waypoints)
        prev_idx = (closest_idx - 1) % n_waypoints
        next_idx = (closest_idx + 1) % n_waypoints
        
        # Compute distance to each segment
        segments = [
            (prev_idx, closest_idx),
            (closest_idx, next_idx)
        ]
        
        min_dist = float('inf')
        for idx1, idx2 in segments:
            p1 = self.waypoints[idx1]
            p2 = self.waypoints[idx2]
            
            # Vector from p1 to p2
            segment_vec = p2 - p1
            segment_len_sq = np.dot(segment_vec, segment_vec)
            
            if segment_len_sq < 1e-6:  # Segment is too short
                # Just use point-to-point distance
                dist = np.linalg.norm(car_pos - p1)
            else:
                # Vector from p1 to car
                car_vec = car_pos - p1
                
                # Project car position onto segment
                t = np.clip(np.dot(car_vec, segment_vec) / segment_len_sq, 0.0, 1.0)
                proj_point = p1 + t * segment_vec
                
                # Perpendicular distance from car to segment
                dist = np.linalg.norm(car_pos - proj_point)
            
            min_dist = min(min_dist, dist)
        
        return float(min_dist)
        
    def reset(self, poses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Resets the F1Tenth simulator, initializes episode tracking variables,
        and returns the initial normalized observation.
        
        Args:
            poses: Optional initial pose array [[x, y, theta]]. If None, uses
                   environment's default starting position.
        
        Returns:
            Normalized lidar observation as numpy array (shape: (scan_beams,))
            with values in [0, 1] range.
        """
        if poses is None:
            # Use the environment's default starting position if available
            # This is typically x=-1.0, y=0.0, theta=0.0 for Spielberg map
            if hasattr(self.env, 'start_xs') and len(self.env.start_xs) > 0:
                poses = np.array([[self.env.start_xs[0], self.env.start_ys[0], self.env.start_thetas[0]]])
            else:
                poses = np.array([[0.0, 0.0, 0.0]])
        
        obs, _, done, info = self.env.reset(poses)
        # Initialize prev_progress and prev position from actual starting position
        x = obs['poses_x'][0] if 'poses_x' in obs else 0.0
        y = obs['poses_y'][0] if 'poses_y' in obs else 0.0
        self.prev_x = x
        self.prev_y = y
        if self.use_centerline_progress:
            self.prev_progress = self._compute_centerline_progress(x, y)
        else:
            self.prev_progress = x
        self.step_count = 0
        self.prev_steer = 0.0
        self.prev_prev_steer = 0.0  # Track steering history for oscillation detection
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        # Track cumulative reward components for episode breakdown
        self.episode_reward_components = {
            'progress': 0.0,
            'speed': 0.0,
            'safety': 0.0,
            'center': 0.0,
            'turning': 0.0,
            'smooth': 0.0,
            'crash': 0.0,
            'total': 0.0
        }
        # Reset episode done flag
        self._episode_done = False
        self._last_obs = None
        self._last_info = {}
        
        normalized_obs = self._normalize_obs(obs['scans'][0])
        self._last_obs = normalized_obs
        return normalized_obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Takes an action (steering, speed), executes it in the simulator,
        computes shaped reward, detects crashes, and returns observation/reward/done/info.
        
        Crash Detection:
        - Checks if environment's done flag is True AND
        - Either original_reward < 0 (environment collision signal) OR
        - Minimum lidar distance < 0.3 meters (very close to obstacle)
        
        Episode Termination:
        - Crash detected (large penalty: -35,000)
        - Time limit reached (5 minutes default)
        
        Args:
            action: Array [steering_angle, speed]
                   - steering_angle: radians, clipped to [-0.4189, 0.4189]
                   - speed: m/s, clipped to [min_speed, max_speed]
        
        Returns:
            Tuple of:
            - observation: Normalized lidar scan (shape: (scan_beams,), dtype: float32)
            - reward: Shaped reward (sum of all reward components)
            - done: True if episode ended (crash or time limit)
            - info: Dictionary containing:
                - reward_components: Per-step reward breakdown
                - episode_reward_components: Cumulative episode reward breakdown
                - episode_reward: Total episode reward
                - episode_duration: Episode duration in seconds
                - crashed: Boolean indicating if episode ended in crash
                - time_limit_reached: Boolean indicating if time limit was reached
        """
        if hasattr(self, '_episode_done') and self._episode_done:
            return self._last_obs, 0.0, True, self._last_info
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        obs, original_reward, done, info = self.env.step(np.array([action]))
        
        self.step_count += 1
        elapsed = self.step_count * self.sim_dt
        time_limit_reached = elapsed >= self.max_episode_seconds
        
        normalized_obs = self._normalize_obs(obs['scans'][0])
        
        shaped_reward, reward_components = self._compute_reward(original_reward, obs, done, info, action)
        
        # Crash detection: Check if car collided with wall/obstacle
        crash_penalty = 0.0
        min_distance = np.min(obs['scans'][0])  # Raw lidar distance in meters
        # Crash if: episode done AND (negative reward OR very close to obstacle)
        crashed = bool(done and (original_reward < 0 or min_distance < 0.3))
        if crashed:
            crash_penalty = -35000.0  # Large penalty for crashing
            shaped_reward += crash_penalty
            reward_components['crash'] = crash_penalty
            reward_components['total'] = shaped_reward
        else:
            reward_components['crash'] = 0.0
        
        self.episode_reward += shaped_reward
        self.episode_reward_components['progress'] += reward_components['progress']
        self.episode_reward_components['speed'] += reward_components['speed']
        self.episode_reward_components['safety'] += reward_components['safety']
        self.episode_reward_components['center'] += reward_components['center']
        self.episode_reward_components['turning'] += reward_components['turning']
        self.episode_reward_components['smooth'] += reward_components['smooth']
        self.episode_reward_components['crash'] += reward_components.get('crash', 0.0)
        self.episode_reward_components['total'] += reward_components['total']
        
        # Store reward components in info for logging
        info = dict(info or {})
        info['reward_components'] = reward_components
        
        # Check if crashed (already computed above with crash penalty)
        # Episodes end on crash or time limit (3 minutes)
        final_done = bool(done or time_limit_reached)
        
        # Store done state to prevent further steps
        self._episode_done = final_done
        
        if final_done:
            info = dict(info or {})
            info['reward_components'] = reward_components
            info['episode_reward_components'] = self.episode_reward_components.copy()
            info['episode_reward'] = self.episode_reward
            info['episode_duration'] = elapsed
            info['crashed'] = crashed
            info['time_limit_reached'] = time_limit_reached and not crashed
            self._last_info = info
            
            crash_status = "CRASHED" if crashed else "TIME LIMIT"
            time_before_crash = elapsed if crashed else "N/A"
            steps_in_episode = self.step_count
            print(f"[Episode End] Total Reward: {self.episode_reward:.2f}, Duration: {elapsed:.2f}s, Steps: {steps_in_episode}, Status: {crash_status}")
            print(f"  Reward breakdown (cumulative): Progress={self.episode_reward_components['progress']:.3f}, "
                  f"Speed={self.episode_reward_components['speed']:.3f}, "
                  f"Safety={self.episode_reward_components['safety']:.3f}, "
                  f"Center={self.episode_reward_components['center']:.3f}, "
                  f"Turning={self.episode_reward_components['turning']:.3f}, "
                  f"Smooth={self.episode_reward_components['smooth']:.3f}, "
                  f"Crash={self.episode_reward_components['crash']:.3f}, "
                  f"Total={self.episode_reward_components['total']:.3f}")
        
        self._last_obs = normalized_obs
        
        return normalized_obs, shaped_reward, final_done, info
    
    def _normalize_obs(self, scan: np.ndarray) -> np.ndarray:
        """
        Normalize lidar scan to [0, 1] range.
        
        Divides raw lidar distances (0-30m) by max_range (30m) and clips to [0, 1].
        This normalization is important for neural network training stability.
        
        Args:
            scan: Raw lidar scan array in meters (0-30m range)
        
        Returns:
            Normalized scan array with values in [0, 1] range, dtype float32
        """
        normalized = np.clip(scan / self.max_range, 0.0, 1.0)
        return normalized.astype(np.float32)
    
    def _compute_reward(self, original_reward: float, obs: Dict, done: bool, info: Dict, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward with multiple components.
        
        Combines several reward components to guide the agent toward desired behavior:
        - Progress: Rewards forward movement (primary driver)
        - Speed: Encourages maintaining forward velocity
        - Safety: Penalizes being too close to walls, rewards safe distance
        - Centering: Keeps car in track center using lidar asymmetry and waypoints
        - Turning: Small penalty for turning (allows necessary turns)
        - Smoothness: Penalizes erratic steering and oscillations
        
        Reward Component Ratios (target):
        - Progress: 3 parts (multiplier: 15.0 per meter)
        - Safety: 2 parts (penalty: -2.0 if <0.2m, reward: +2.0 if safe)
        - Speed: 0.5 parts (multiplier: 0.625 per m/s)
        - Center: 0.5 parts (scaled to max 0.5)
        - Turning: 0.5 parts (max penalty: -0.5)
        - Smooth: 0.5 parts (various penalties)
        
        Args:
            original_reward: Original reward from F1Tenth environment (not used)
            obs: Observation dictionary containing:
                - scans: Lidar scan array
                - poses_x, poses_y: Car position
                - linear_vels_x: Forward velocity
            done: Whether episode is done (from environment)
            info: Info dictionary from environment
            action: Action array [steering_angle, speed]
        
        Returns:
            Tuple of:
            - total_reward: Sum of all reward components
            - reward_components: Dictionary with individual component values:
                - progress: Forward movement reward
                - speed: Speed reward
                - safety: Safety reward/penalty
                - center: Centering reward (scaled)
                - center_original: Centering reward (before scaling)
                - turning: Turning penalty
                - smooth: Smoothness penalty
                - total: Total reward (same as first return value)
        """
        scan = obs['scans'][0]  # Raw lidar scan (0-30m range)
        
        x = obs['poses_x'][0]
        y = obs['poses_y'][0] if 'poses_y' in obs else 0.0
        
        # Progress reward: Calculate forward movement
        # Method 1 (preferred): Euclidean distance traveled (if waypoints available)
        if self.use_centerline_progress and hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            dx = x - self.prev_x
            dy = y - self.prev_y
            distance_traveled = np.sqrt(dx*dx + dy*dy)
            progress_delta = distance_traveled
            current_progress = None
        else:
            # Method 2: Centerline progress or x-coordinate
            if self.use_centerline_progress:
                current_progress = self._compute_centerline_progress(x, y)
                progress_delta = current_progress - self.prev_progress
            else:
                current_progress = x
                progress_delta = current_progress - self.prev_progress
            
            # Handle wrap-around (if progress jumps backward by >50% of track)
            if (progress_delta < 0 and self.use_centerline_progress and 
                self.waypoint_distances is not None and len(self.waypoint_distances) > 0):
                track_length = self.waypoint_distances[-1]
                if abs(progress_delta) > track_length * 0.5:
                    progress_delta = 0.0
        
        # Update previous position
        self.prev_x = x
        self.prev_y = y
        
        if current_progress is not None:
            self.prev_progress = current_progress
        
        # Safety checks: filter out teleports/resets and backward movement
        max_progress_per_step = 0.05  # Maximum reasonable progress per step (5cm)
        if abs(progress_delta) > max_progress_per_step:
            progress_delta = 0.0  # Likely a reset/teleport
        elif progress_delta < 0.0:
            progress_delta = 0.0  # No reward for going backward
        
        progress_reward = progress_delta * 15.0  # 15.0 per meter (ratio: 3)
        
        # Speed reward: Encourage forward velocity
        speed = obs['linear_vels_x'][0]
        speed_reward = speed * 0.625  # 0.625 per m/s (ratio: 0.5)
        
        # Safety reward: Penalize being too close to walls, reward safe distance
        min_distance = np.min(scan) * self.max_range  # Convert normalized to meters
        safety_threshold = 0.2  # 20cm threshold
        
        if min_distance < safety_threshold:
            # Penalty increases as distance decreases (max: -2.0 at 0m)
            safety_reward = -2.0 * (1.0 - min_distance / safety_threshold)
        else:
            # Reward for maintaining safe distance (0.2-0.7m range, max: +2.0)
            safety_reward = 2.0 * min(1.0, (min_distance - safety_threshold) / 0.5)
        
        # Centering reward: Keep car in track center
        x = obs['poses_x'][0]
        y = obs['poses_y'][0] if 'poses_y' in obs else 0.0
        
        # Get lidar readings at ±90 degrees (left and right sides)
        left_scan_window = scan[self.left_90_idx_start:self.left_90_idx_end+1]
        right_scan_window = scan[self.right_90_idx_start:self.right_90_idx_end+1]
        left_wall_distance = float(np.mean(left_scan_window)) * self.max_range
        right_wall_distance = float(np.mean(right_scan_window)) * self.max_range
        
        # Compute distance to centerline if waypoints available
        centerline_distance = 0.0
        if self.use_centerline_progress and self.waypoints is not None:
            centerline_distance = self._compute_distance_to_centerline(x, y)
        
        avg_wall_distance = (left_wall_distance + right_wall_distance) / 2.0
        
        if avg_wall_distance > 0.1:  # Only compute if walls are detected
            # Lidar-based centering: reward symmetric wall distances
            wall_distance_diff = abs(left_wall_distance - right_wall_distance)
            normalized_asymmetry = wall_distance_diff / avg_wall_distance
            lidar_center_reward = 2.0 * (1.0 - normalized_asymmetry) ** 2  # Max +2.0 when perfectly centered
            
            # Waypoint-based centering: reward being on centerline
            if self.use_centerline_progress:
                estimated_track_width = avg_wall_distance * 2.0
                if estimated_track_width > 0.5:
                    centerline_normalized = min(centerline_distance / (estimated_track_width / 2.0), 1.0)
                    centerline_center_reward = 2.0 * (1.0 - centerline_normalized) ** 2
                else:
                    centerline_center_reward = 0.0
            else:
                centerline_center_reward = 0.0
            
            # Combine both methods if waypoints available, else use lidar only
            center_reward = (lidar_center_reward + centerline_center_reward) / 2.0 if self.use_centerline_progress else lidar_center_reward
        else:
            center_reward = 0.0
        
        # Scale centering reward to target ratio (0.5)
        center_reward_scaled = center_reward * 0.25  # Scale to max +0.5
        center_reward_max_magnitude = 0.5
        
        # Smoothness penalties: Encourage smooth, predictable driving
        steer = float(action[0]) if action.ndim == 1 else float(action[0][0])
        steer_change = abs(steer - self.prev_steer)
        
        # Turning penalty: Small quadratic penalty for turning (allows necessary turns)
        steer_magnitude = abs(steer)
        max_steering_angle = 0.4189  # ±24 degrees
        turning_penalty = -0.5 * (steer_magnitude / max_steering_angle) ** 2  # Max -0.5 at full steering
        
        # Steering magnitude penalty: Penalize large steering angles
        steer_magnitude_penalty = -0.05 * max(0.0, abs(steer) - 0.05)
        
        # Steering change penalty: Penalize large steering changes
        steer_change_penalty = -0.05 * max(0.0, steer_change - 0.01)
        
        # Oscillation detection: Detect zigzag patterns (left-right-left or right-left-right)
        oscillation_penalty = 0.0
        
        # Detect sign changes in steering (oscillation)
        if abs(self.prev_steer) > 0.02 and abs(steer) > 0.02:
            if ((self.prev_steer > 0 and steer < 0) or (self.prev_steer < 0 and steer > 0)):
                oscillation_penalty = -0.2 * min(steer_change / 0.4, 2.0)
        
        # Detect three-step zigzag pattern
        if abs(self.prev_prev_steer) > 0.02 and abs(self.prev_steer) > 0.02 and abs(steer) > 0.02:
            if ((self.prev_prev_steer > 0 and self.prev_steer < 0 and steer > 0) or
                (self.prev_prev_steer < 0 and self.prev_steer > 0 and steer < 0)):
                oscillation_penalty = -0.2
        
        smooth_penalty = steer_magnitude_penalty + steer_change_penalty + oscillation_penalty
        
        self.prev_prev_steer = self.prev_steer
        self.prev_steer = steer
        
        total_reward = progress_reward + speed_reward + safety_reward + center_reward_scaled + turning_penalty + smooth_penalty
        reward_components = {
            'progress': progress_reward,
            'speed': speed_reward,
            'safety': safety_reward,
            'center': center_reward_scaled,
            'center_original': center_reward,
            'turning': turning_penalty,
            'smooth': smooth_penalty,
            'total': total_reward
        }
        
        return total_reward, reward_components
    
    def render(self, mode='human'):
        """
        Render the environment visualization.
        
        Args:
            mode: Rendering mode ('human' for display, 'rgb_array' for image)
        
        Returns:
            Rendered frame (if mode='rgb_array') or None
        """
        return self.env.render(mode=mode)
    
    def close(self):
        """
        Close the environment and clean up resources.
        
        Should be called when done with the environment to free resources.
        """
        self.env.close()

