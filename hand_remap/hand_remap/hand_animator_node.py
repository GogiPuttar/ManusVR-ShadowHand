import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
import pandas as pd
import numpy as np
import re
import os

class HandAnimator(Node):
    def __init__(self):
        super().__init__('hand_animator')
        self.declare_parameter('csv_path', '')
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        if not os.path.isfile(csv_path):
            self.get_logger().error(f"CSV file not found: {csv_path}")
            return

        self.publisher = self.create_publisher(MarkerArray, 'hand_markers', 10)
        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.frame = 0

        self.node_positions, self.node_orientations = self.load_data(csv_path)
        self.total_frames = self.node_positions.shape[0]
        self.get_logger().info(f"Loaded {self.total_frames} frames")

        self.colors = [ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), # cyan
                       ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), # green
                       ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), # blue
                       ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), # red
                       ]

    def load_data(self, path):
        df = pd.read_csv(path)
        pos_cols = [c for c in df.columns if re.match(r'p\d+_[xyz]', c)]
        node_positions = df[pos_cols].values.reshape(len(df), -1, 3)[::2]
        orient_cols = [c for c in df.columns if re.match(r'o\d+_[xyzw]', c)]
        node_orientations = df[orient_cols].values.reshape(len(df), -1, 4)[::2]
        return node_positions, node_orientations

    def timer_callback(self):
        if self.frame >= self.total_frames:
            self.frame = 0  # Loop

        hand_chains = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                       [13, 14, 15, 16], [17, 18, 19, 20]]
        global_positions = []

        for chain in hand_chains:
            parent_position = np.array([0.0, 0.0, 0.0])
            parent_orientation = np.array([0.0, 0.0, 0.0, 1.0])

            for idx in chain:
                q = self.node_orientations[self.frame][idx]
                p = self.node_positions[self.frame][idx]
                orientation = self.quaternion_multiply(parent_orientation, q)
                position = parent_position + self.quaternion_rotate(p, parent_orientation)
                global_positions.append(position)
                parent_position = position
                parent_orientation = orientation

        marker_array = MarkerArray()
        for i, pos in enumerate(global_positions):
            marker = Marker()
            marker.header = Header(frame_id='world')
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0  # identity
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color = self.colors[i % len(self.colors)]
            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
        self.frame += 1

    def quaternion_rotate(self, v, q):
        q = np.array([q[0], q[1], q[2], q[3]])
        v = np.array([v[0], v[1], v[2], 0.0])
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        v_rot = self.quaternion_multiply(self.quaternion_multiply(q, v), q_conj)
        return v_rot[0:3]

    def quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

def main(args=None):
    rclpy.init(args=args)
    node = HandAnimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
