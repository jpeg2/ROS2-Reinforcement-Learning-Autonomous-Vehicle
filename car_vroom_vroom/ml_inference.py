#!/usr/bin/env python3
"""
ROS2 node for running trained ML model inference on F1Tenth car.
Subscribes to sensor data and publishes control commands.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import torch
from stable_baselines3 import PPO
import os


class MLInferenceNode(Node):
    """
    ROS2 node that runs ML inference using a trained model.
    """
    
    def __init__(self):
        super().__init__('ml_inference')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', 'gf_to_aeb')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('device', 'auto')
        self.declare_parameter('min_speed', 0.5)
        self.declare_parameter('max_speed', 5.0)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.max_range = self.get_parameter('max_range').value
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        
        # Device selection
        device_param = self.get_parameter('device').value
        if device_param == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_param
        
        self.get_logger().info(f'Using device: {self.device}')
        
        # Load model
        if not model_path or not os.path.exists(model_path):
            self.get_logger().error(f'Model path not found: {model_path}')
            raise ValueError(f'Model path not found: {model_path}')
        
        self.get_logger().info(f'Loading model from: {model_path}')
        try:
            self.model = PPO.load(model_path, device=self.device)
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise
        
        # State variables
        self.last_scan = None
        self.last_odom = None
        
        # Create subscribers (using default QoS: RELIABLE, KEEP_LAST, depth=10, VOLATILE)
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        # Create publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )
        
        # Timer for inference (run at 20 Hz)
        self.inference_timer = self.create_timer(0.05, self.inference_callback)
        
        self.get_logger().info('ML Inference node started')
    
    def scan_callback(self, msg: LaserScan):
        """Store latest scan data."""
        # Convert to numpy array and normalize
        ranges = np.array(msg.ranges)
        # Replace inf/nan with max_range
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        # Normalize to [0, 1]
        normalized_ranges = np.clip(ranges / self.max_range, 0.0, 1.0)
        self.last_scan = normalized_ranges.astype(np.float32)
    
    def odom_callback(self, msg: Odometry):
        """Store latest odometry data."""
        self.last_odom = msg
    
    def inference_callback(self):
        """Run ML inference and publish control commands."""
        if self.last_scan is None:
            return
        
        try:
            # The model expects normalized observations with shape (1080,)
            # Ensure we have exactly 1080 beams (trim if needed)
            scan = self.last_scan.flatten()
            if len(scan) != 1080:
                # If scan size doesn't match, pad or truncate
                if len(scan) > 1080:
                    scan = scan[:1080]
                else:
                    # Pad with max_range if too short
                    scan = np.pad(scan, (0, 1080 - len(scan)), mode='constant', constant_values=1.0)
            
            observation = scan.astype(np.float32)
            action, _ = self.model.predict(observation, deterministic=True)
            
            # action is [steering_angle, speed] - handle both single and batched outputs
            if action.ndim > 1:
                steering_angle = float(action[0][0])
                speed = float(action[0][1])
            else:
                steering_angle = float(action[0])
                speed = float(action[1])
            
            # Clamp to action limits and apply speed floor/ceiling
            steering_angle = np.clip(steering_angle, -0.4189, 0.4189)
            speed = np.clip(speed, self.min_speed, self.max_speed)
            
            # Publish control command
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = -steering_angle
            
            # Log what we're publishing
            self.get_logger().info(f"Publishing drive: speed={speed:.2f} m/s, steer={-steering_angle:.3f} rad")
            
            self.drive_pub.publish(drive_msg)
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = MLInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

