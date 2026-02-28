import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


class SafetyNode(Node):

    """
    A ROS2 node that implements a simple autonomous emergency braking (AEB) system.

    Subscriptions:
    'gf_to_aeb' (AckermannDriveStamped): Drive commands from the gap-following logic.
    '/scan' (LaserScan): LiDAR scan data for obstacle detection.

    Publications:
    '/drive' (AckermannDriveStamped): Final, safety-checked drive commands.

    Parameters:
    stop_distance (float):      Base distance threshold (in meters) to trigger emergency braking.
    angle_damping_deg (float):  Angular range (in degrees) from the center where damping starts.
                                Beyond this angle, thresholds can be reduced for side clearance.
    damping_factor (float):     Multiplier applied to stop_distance in side regions (0–1).
                                For example, 0.5 means reduce the stop distance by 50%.
    shut_speed (float):         The velocity (usually 0.0) applied when braking is active.
    """
    def __init__(self):

        
        super().__init__('safety_node')

        self.declare_parameter('stop_distance', 0.15)        # base stopping distance
        self.declare_parameter('angle_damping_deg', 45.0)   # side angle to start damping
        self.declare_parameter('damping_factor', 0.5)       # 50% threshold
        self.declare_parameter('shut_speed', 0.0)           # stop speed

        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.angle_damping = math.radians(float(self.get_parameter('angle_damping_deg').value))
        self.damping_factor = float(self.get_parameter('damping_factor').value)
        self.shut_speed = float(self.get_parameter('shut_speed').value)

        self.current_steering = 0.0
        self.brake_active = False

        # create subscriptions to drive logic, safety node makes final decision on speeds and whether to stop
        self.create_subscription(AckermannDriveStamped, 'gf_to_aeb', self.drive_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info("SafetyNode started")


    def drive_callback(self, msg: AckermannDriveStamped):
        """
        Callback for drive commands from the navigation logic.

        If AEB is not active, the command is forwarded to the drive topic.
        If AEB is active, the command is overridden to a full stop (shut_speed).
        """
        if not self.brake_active:
            self.drive_publisher.publish(msg)
        else:
            brake_msg = AckermannDriveStamped()
            brake_msg.drive.speed = self.shut_speed
            self.drive_publisher.publish(brake_msg)

    # gets ranges from lidar
    def scan_callback(self, msg: LaserScan):
        """
        Callback for incoming LiDAR scan data.

        This method checks if any obstacle is within the stop distance threshold.
        If an obstacle is detected, braking is activated and further drive commands
        are suppressed until the node is reset.
        """
        ranges = np.array(msg.ranges, dtype=float)

        # filters out invalid readings
        ranges[np.isnan(ranges)] = float('inf')
        ranges[np.isinf(ranges)] = float('inf')

        # calculate angles and relative distances
        for i, r in enumerate(ranges):
            angle = msg.angle_min + i * msg.angle_increment

            threshold = self.stop_distance

            # Steering left -> damp right side
            if self.current_steering > 0 and angle < -self.angle_damping:
                threshold *= self.damping_factor

            # Steering right -> damp left side
            elif self.current_steering < 0 and angle > self.angle_damping:
                threshold *= self.damping_factor

            if r < threshold:
                self.brake_active = True
                brake_msg = AckermannDriveStamped()
                brake_msg.drive.speed = self.shut_speed
                self.drive_publisher.publish(brake_msg)
                self.get_logger().warn("AEB engaged, brake active")
                break


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
