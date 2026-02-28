"""
Wrapper for F1Tenth Gym environment to work with RL algorithms.
"""
import gym
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional


class F1TenthEnvWrapper(gym.Env):
    """
    Wrapper for F1Tenth gym environment that provides:
    - Proper observation and action spaces
    - Reward shaping for training
    - State normalization
    - Episode termination on time limit
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
    
    def _load_waypoints(self, waypoints_path: str):
        """Load centerline waypoints from CSV file."""
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
        This is more stable than just using the closest waypoint.
        
        Returns:
            Cumulative distance along centerline (or x-coordinate if no waypoints)
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
        Returns positive distance in meters (0 = on centerline, positive = off centerline).
        
        Returns:
            Distance from car to centerline in meters, or 0.0 if no waypoints available
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
        """Reset the environment."""
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
        """Take a step in the environment."""
        # If already done (from previous step), don't take another step
        # This prevents the environment from continuing after a crash
        if hasattr(self, '_episode_done') and self._episode_done:
            # Return the last observation with done=True to signal reset needed
            return self._last_obs, 0.0, True, self._last_info
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        obs, original_reward, done, info = self.env.step(np.array([action]))
        
        # Time accumulation
        self.step_count += 1
        elapsed = self.step_count * self.sim_dt
        time_limit_reached = elapsed >= self.max_episode_seconds
        
        # Normalize observation
        normalized_obs = self._normalize_obs(obs['scans'][0])
        
        # Compute shaped reward and components
        shaped_reward, reward_components = self._compute_reward(original_reward, obs, done, info, action)
        
        # Add crash penalty if crashed
        crash_penalty = 0.0
        min_distance = np.min(obs['scans'][0])
        crashed = bool(done and (original_reward < 0 or min_distance < 0.3))
        if crashed:
            crash_penalty = -35000.0  # Extremely large penalty for crashing
            shaped_reward += crash_penalty
            reward_components['crash'] = crash_penalty
            reward_components['total'] = shaped_reward  # Update total to include crash penalty
        else:
            reward_components['crash'] = 0.0
        
        # Track episode reward
        self.episode_reward += shaped_reward
        
        # Accumulate reward components for episode breakdown
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
            # Create new info dict, preserving reward_components
            info = dict(info or {})
            info['reward_components'] = reward_components  # Always include latest reward components
            info['episode_reward'] = self.episode_reward
            info['episode_duration'] = elapsed
            info['crashed'] = crashed
            info['time_limit_reached'] = time_limit_reached and not crashed
            self._last_info = info
            
            # Print episode statistics with cumulative reward breakdown
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
        
        # Store last observation for potential repeat calls after done
        self._last_obs = normalized_obs
        
        return normalized_obs, shaped_reward, final_done, info
    
    def _normalize_obs(self, scan: np.ndarray) -> np.ndarray:
        """Normalize lidar scan to [0, 1] range."""
        normalized = np.clip(scan / self.max_range, 0.0, 1.0)
        return normalized.astype(np.float32)
    
    def _compute_reward(self, original_reward: float, obs: Dict, done: bool, info: Dict, action: np.ndarray) -> float:
        """
        Reward components (no hard-coded track widths):
        - Progress: forward movement reward
        - Speed: reward for maintaining speed
        - Safety: penalize being closer than 0.2m to walls, reward safe distance
        - Centering: use left/right lidar at ±90 degrees to keep car centered
        - Smoothness: penalize excessive steering and oscillations
        """
        scan = obs['scans'][0]
        
        # Progress reward - use actual distance traveled along centerline if waypoints available, else x-coordinate
        # Calculate progress based on actual movement (more reliable than waypoint jumping)
        x = obs['poses_x'][0]
        y = obs['poses_y'][0] if 'poses_y' in obs else 0.0
        
        # Calculate progress based on actual distance traveled (more reliable than waypoint jumping)
        if self.use_centerline_progress and hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            # Calculate Euclidean distance traveled
            dx = x - self.prev_x
            dy = y - self.prev_y
            distance_traveled = np.sqrt(dx*dx + dy*dy)
            
            # Use distance traveled as progress (works well for small steps)
            progress_delta = distance_traveled
            current_progress = None  # Not needed when using distance-based tracking
        else:
            # Fallback: use x-coordinate or waypoint-based (if first step or no waypoints)
            if self.use_centerline_progress:
                current_progress = self._compute_centerline_progress(x, y)
                progress_delta = current_progress - self.prev_progress
            else:
                current_progress = x
                progress_delta = current_progress - self.prev_progress
            
            # Handle wrap-around for closed-loop tracks
            if (progress_delta < 0 and self.use_centerline_progress and 
                self.waypoint_distances is not None and len(self.waypoint_distances) > 0):
                track_length = self.waypoint_distances[-1]
                if abs(progress_delta) > track_length * 0.5:
                    # Wrap-around: car went from end of track to start
                    progress_delta = 0.0
        
        # Store current position for next step
        self.prev_x = x
        self.prev_y = y
        
        # Update prev_progress if using waypoint-based tracking (not distance-based)
        if current_progress is not None:
            self.prev_progress = current_progress
        
        # Clamp progress_delta to reasonable maximum per step
        # At 0.8 m/s max speed with 0.01s timestep, max distance per step is ~0.008m
        # Allow margin for measurement error: clamp to 0.05m per step
        max_progress_per_step = 0.05  # Maximum meters per step (sanity check)
        if abs(progress_delta) > max_progress_per_step:
            # Unrealistic jump - likely a bug, ignore it
            progress_delta = 0.0
        elif progress_delta < 0.0:
            # Small negative delta - likely measurement noise or slight backtracking, ignore it
            progress_delta = 0.0
        
        # Reward scaling to match target ratios: Progress=3.0, Speed=0.5, Safety=2.0, Center=0.5, Turning=-0.5, Smooth=-0.5
        # At 0.8 m/s with 0.01s timesteps: ~0.008m per step max, clamp to 0.2m per step = max 20.0 per step
        # Scale progress to target ratio of 3.0: progress_reward = progress_delta * (3.0 / 0.2) = 15.0 per meter
        # But max per step is 0.2m, so max progress per step = 15.0 * 0.2 = 3.0 ✓
        progress_reward = progress_delta * 100.0  # Scaled to max 3.0 per step (ratio 3.0)
        # Note: self.prev_progress is updated earlier if using waypoint-based tracking
        
        # Speed reward - scaled to ratio 0.5 (target max 0.5 per step)
        # At max speed 0.8 m/s, we want 0.5 reward, so scale = 0.5 / 0.8 = 0.625
        speed = obs['linear_vels_x'][0]
        speed_reward = speed * 0.625  # Scaled to max 0.5 per step at 0.8 m/s (ratio 0.5)
        
        # Safety reward - always stay 0.2m away from closest wall (no track width dependency)
        min_distance = np.min(scan) * self.max_range  # Convert to meters
        safety_threshold = 0.2  # Always require 0.2m clearance
        
        # Safety reward - scaled to ratio 2.0 (target max +2.0 per step when safe, max -2.0 when unsafe)
        if min_distance < safety_threshold:
            # Penalty for being too close to walls (< 0.2m)
            # Penalty increases as distance decreases
            # Scale from max -5.0 to max -2.0: multiply by 0.4
            safety_reward = -2.0 * (1.0 - min_distance / safety_threshold)  # -2.0 when at wall, 0.0 at 0.2m (ratio 2.0)
        else:
            # Reward for maintaining safe distance (>= 0.2m)
            # Scale from max +0.1 to max +2.0: multiply by 20.0
            safety_reward = 2.0 * min(1.0, (min_distance - safety_threshold) / 0.5)  # Max +2.0 at 0.7m+ (ratio 2.0)
        
        # Centering: combine centerline distance with lidar-based wall distances
        # Get car position
        x = obs['poses_x'][0]
        y = obs['poses_y'][0] if 'poses_y' in obs else 0.0
        
        # Get lidar readings at ±90 degrees (left and right walls)
        left_scan_window = scan[self.left_90_idx_start:self.left_90_idx_end+1]
        right_scan_window = scan[self.right_90_idx_start:self.right_90_idx_end+1]
        left_wall_distance = float(np.mean(left_scan_window)) * self.max_range  # Convert to meters
        right_wall_distance = float(np.mean(right_scan_window)) * self.max_range  # Convert to meters
        
        # Compute distance from car to centerline (if waypoints available)
        centerline_distance = 0.0
        if self.use_centerline_progress and self.waypoints is not None:
            centerline_distance = self._compute_distance_to_centerline(x, y)
        
        # Calculate centering reward using both methods:
        # 1. Lidar asymmetry: reward when left ≈ right (car is equidistant from walls)
        # 2. Centerline distance: reward when car is close to centerline
        avg_wall_distance = (left_wall_distance + right_wall_distance) / 2.0
        
        if avg_wall_distance > 0.1:  # Avoid division by very small numbers
            # Normalized asymmetry: 0 = perfectly centered between walls, 1 = very off-center
            wall_distance_diff = abs(left_wall_distance - right_wall_distance)
            normalized_asymmetry = wall_distance_diff / avg_wall_distance
            
            # Lidar-based centering reward (when left = right, asymmetry = 0)
            lidar_center_reward = 2.0 * (1.0 - normalized_asymmetry) ** 2  # Max +2.0 when perfectly centered
            
            # Centerline-based reward (penalize distance from centerline)
            # Assume track width of ~2m, reward decreases quadratically with distance
            # Max reward when centerline_distance = 0, zero reward at ~1m away
            if self.use_centerline_progress:
                # Use track width estimate from average wall distance
                estimated_track_width = avg_wall_distance * 2.0  # Track width ≈ 2 * avg distance to wall
                if estimated_track_width > 0.5:  # Valid track width estimate
                    centerline_normalized = min(centerline_distance / (estimated_track_width / 2.0), 1.0)
                    centerline_center_reward = 2.0 * (1.0 - centerline_normalized) ** 2
                else:
                    centerline_center_reward = 0.0
            else:
                centerline_center_reward = 0.0
            
            # Combine both rewards (average them for balanced approach)
            center_reward = (lidar_center_reward + centerline_center_reward) / 2.0 if self.use_centerline_progress else lidar_center_reward
        else:
            # Too close to walls, no centering reward
            center_reward = 0.0
        
        # Scale centering reward to target ratio 0.5 (target max +0.5 per step)
        # Original reward range is [0, 2.0] for perfectly centered
        # Scale to max +0.5: multiply by 0.25
        center_reward_scaled = center_reward * 0.25  # Scale to max +0.5 (ratio 0.5)
        center_reward_max_magnitude = 0.5  # Maximum centering reward magnitude
        
        # Extract steering angle (used for turning penalty and smoothness)
        steer = float(action[0]) if action.ndim == 1 else float(action[0][0])
        steer_change = abs(steer - self.prev_steer)
        
        # Turning penalty: scaled to target ratio -0.5 (target max -0.5 per step at full steering)
        # Max steering is ±0.4189 rad (~24 degrees), so normalize by that
        steer_magnitude = abs(steer)
        max_steering_angle = 0.4189  # Maximum steering angle in radians
        # Penalty scales quadratically with steering: small penalty for gentle turns, larger for sharp turns
        # Scale from max -5.0 to max -0.5: multiply by 0.1
        # Using quadratic scaling so small steering angles have minimal penalty
        turning_penalty = -0.5 * (steer_magnitude / max_steering_angle) ** 2  # Max -0.5 at full steering (ratio 0.5)
        
        # Smoothness: lightly penalize steering changes and oscillations to reduce wiggling
        # Much reduced penalties to allow necessary steering while discouraging excessive wiggling
        
        # Smoothness penalties - scaled to target ratio -0.5 total (max -0.5 per step total)
        # Allow small steering (0.05 rad ≈ 3 degrees) without penalty
        # Scale from max -0.5 to max -0.05: multiply by 0.1
        steer_magnitude_penalty = -0.05 * max(0.0, abs(steer) - 0.05)
        
        # Small penalty for steering changes - encourage smooth steering changes
        # Allow small changes (0.01 rad) without penalty
        # Scale from max -0.5 to max -0.05: multiply by 0.1
        steer_change_penalty = -0.05 * max(0.0, steer_change - 0.01)
        
        # Detect and lightly penalize oscillations (rapid direction changes)
        oscillation_penalty = 0.0
        
        # Penalize steering direction changes (sign flip) - indicates oscillation/wiggling
        if abs(self.prev_steer) > 0.02 and abs(steer) > 0.02:  # Need significant steering for both
            # Check if steering direction changed (sign flip)
            if ((self.prev_steer > 0 and steer < 0) or (self.prev_steer < 0 and steer > 0)):
                # Sign flip indicates oscillation - scale from max -2.0 to max -0.2: multiply by 0.1
                oscillation_penalty = -0.2 * min(steer_change / 0.4, 2.0)  # Max -0.2 for large flips
        
        # Also check for zigzag patterns (left -> right -> left)
        if abs(self.prev_prev_steer) > 0.02 and abs(self.prev_steer) > 0.02 and abs(steer) > 0.02:
            # Check for zigzag: left -> right -> left or right -> left -> right
            if ((self.prev_prev_steer > 0 and self.prev_steer < 0 and steer > 0) or
                (self.prev_prev_steer < 0 and self.prev_steer > 0 and steer < 0)):
                # Scale from max -2.0 to max -0.2: multiply by 0.1
                oscillation_penalty = -0.2  # Max total smooth penalty is ~-0.5 (-0.05 + -0.05 + -0.2 = -0.3, which is < 0.5)
        
        smooth_penalty = steer_magnitude_penalty + steer_change_penalty + oscillation_penalty
        
        # Update steering history
        self.prev_prev_steer = self.prev_steer
        self.prev_steer = steer
        
        # Calculate total reward
        total_reward = progress_reward + speed_reward + safety_reward + center_reward_scaled + turning_penalty + smooth_penalty
        
        # Store reward components for logging
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
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()

