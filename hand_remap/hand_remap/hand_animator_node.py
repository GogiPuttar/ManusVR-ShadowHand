import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header, ColorRGBA
import pandas as pd
import numpy as np
import re
import os
from geometry_msgs.msg import TransformStamped
import yaml
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import PyKDL
import rclpy
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from lxml import etree
import tf_transformations
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster, TransformListener, Buffer
from scipy.optimize import minimize

class HandAnimator(Node):
    def __init__(self):
        super().__init__('hand_animator')
        self.declare_parameter('csv_path', '')
        self.declare_parameter('tracking_path', '')
        self.declare_parameter('robot_description', '')

        # Load params
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        if not os.path.isfile(csv_path):
            self.get_logger().error(f"CSV file not found: {csv_path}")
            return
        tracking_path = self.get_parameter('tracking_path').get_parameter_value().string_value
        if not os.path.isfile(tracking_path):
            self.get_logger().error(f"Tracking params file not found: {tracking_path}")
            return
        with open(tracking_path, 'r') as f:
            tracking_config = yaml.safe_load(f)

        self.robot_description = self.remove_unsupported_tags(self.get_parameter('robot_description').get_parameter_value().string_value)

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

        self.mode = tracking_config.get('mode', 'IK')
        self.scaling_factor = tracking_config.get('scaling_factor', 1.0)
        self.tuning_mode = tracking_config.get('tuning_mode', False)
        self.reference_to_manus = tracking_config['reference_to_manus']

        # Load URDF and parse KDL chains
        self.robot = URDF.from_xml_string(self.robot_description)
        self.log_urdf_tree(self.get_logger())

        success, self.tree = treeFromUrdfModel(self.robot)
        if not success:
            self.get_logger().error("Failed to parse URDF into KDL tree")
            return

        self.kdl_chains = {}
        self.ik_solvers = {}
        self.fk_solvers = {}
        self.fingertip_links = {
            'index': 'rh_fftip',
            'middle': 'rh_mftip',
            'ring': 'rh_rftip',
            'little': 'rh_lftip',
            'thumb': 'rh_thtip',
        }

        for name, tip in self.fingertip_links.items():
            chain = self.tree.getChain('reference', tip)
            self.kdl_chains[name] = chain
            fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
            ik_solver = PyKDL.ChainIkSolverPos_LMA(chain)
            self.ik_solvers[name] = ik_solver
            self.fk_solvers[name] = fk_solver
            self.get_logger().info(f"KDL chain for {name} has {chain.getNrOfJoints()} joints.")

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Broadcast static TF
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf()

    def load_data(self, path):
        df = pd.read_csv(path)
        pos_cols = [c for c in df.columns if re.match(r'p\d+_[xyz]', c)]
        node_positions = df[pos_cols].values.reshape(len(df), -1, 3)[::2]
        orient_cols = [c for c in df.columns if re.match(r'o\d+_[xyzw]', c)]
        node_orientations = df[orient_cols].values.reshape(len(df), -1, 4)[::2]
        return node_positions, node_orientations

    def remove_unsupported_tags(self, urdf_string):
        root = etree.fromstring(urdf_string.encode())
        for tag in root.xpath('//origin_xyz | //ros2_control'):
            tag.getparent().remove(tag)
        return etree.tostring(root).decode()
    
    def log_urdf_tree(self, logger):
        def recurse(link, indent=''):
            children = [j for j in self.robot.joints if j.parent == link]
            for joint in children:
                logger.info(f"{indent}- {joint.child}  (joint: {joint.name}, type: {joint.type})")
                recurse(joint.child, indent + '  ')

        logger.info(f"🤖 URDF Robot: {self.robot.name}")
        logger.info(f"📦 URDF Link Tree:")
        recurse(self.robot.get_root())

    def solve_ik(self, chain_name, pose_in_ref):
        chain = self.kdl_chains[chain_name]
        ik_solver = self.ik_solvers[chain_name]

        # Convert Pose to PyKDL Frame
        position = pose_in_ref.pose.position
        orientation = pose_in_ref.pose.orientation
        frame = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(orientation.x, orientation.y, orientation.z, orientation.w),
            PyKDL.Vector(position.x, position.y, position.z)
        )

        q_init = PyKDL.JntArray(chain.getNrOfJoints())
        q_out = PyKDL.JntArray(chain.getNrOfJoints())

        success = ik_solver.CartToJnt(q_init, frame, q_out)
        if success >= 0:
            return q_out
        else:
            self.get_logger().warn(f"IK failed for {chain_name}")
            return None

    def solve_position_only_ik(self, chain, fk_solver, target_pos, q_seed=None):
        n_joints = chain.getNrOfJoints()

        if q_seed is None:
            q_seed = np.zeros(n_joints)

        def fk_position_error(q):
            q_kdl = PyKDL.JntArray(n_joints)
            for i in range(n_joints):
                q_kdl[i] = q[i]

            frame = PyKDL.Frame()
            fk_solver.JntToCart(q_kdl, frame)
            pos = frame.p
            return np.linalg.norm(np.array([pos[0], pos[1], pos[2]]) - target_pos)

        result = minimize(fk_position_error, q_seed, method='BFGS')
        self.get_logger().info(f"Target: {target_pos}")
        self.get_logger().info(f"Optimized pos error: {fk_position_error(result.x):.4f}")

        final_error = fk_position_error(result.x)
        if result.success or final_error < 1e-4:
            q_result = PyKDL.JntArray(n_joints)
            for i in range(n_joints):
                q_result[i] = result.x[i]
            return q_result
        else:
            self.get_logger().warn(f"IK failed (success={result.success}, error={final_error:.6f})")
            return None

        # if result.success:
        #     q_result = PyKDL.JntArray(n_joints)
        #     for i in range(n_joints):
        #         q_result[i] = result.x[i]
        #     return q_result
        # else:
        #     return None

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
            marker.header = Header(frame_id='manus')
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            if self.tuning_mode:
                pos *= self.scaling_factor

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

        # target = np.array([0.02, -0.03, 0.18])  # desired position in reference frame

        # target = np.array([1.0, 0.0, 0.0])  # desired position in reference frame
        target = np.array([0.04, -0.05, 0.1])  # desired position in reference frame

        # target = np.array([0.02, -0.00, 0.0])  # desired position in reference frame
        # fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)

        q_out_index = self.solve_position_only_ik(self.kdl_chains['index'], self.fk_solvers['index'], target)
        q_out_thumb = self.solve_position_only_ik(self.kdl_chains['thumb'], self.fk_solvers['thumb'], target)

        if q_out_index and q_out_thumb:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()

            name_list = [f"rh_FFJ{i}" for i in range(q_out_index.rows(), 0, -1)] + [f"rh_THJ{i}" for i in range(q_out_thumb.rows(), 0, -1)]
            msg.name = name_list

            # self.get_logger().info(f"{msg.name}")

            pos_list = [q_out_index[i] for i in range(q_out_index.rows())] + [q_out_thumb[i] for i in range(q_out_thumb.rows())]

            msg.position = pos_list
            # self.get_logger().info(f"{msg.position}")

            self.joint_pub.publish(msg)

        # # IK
        # for pos in [global_positions[i] for i in [chain[-1] for chain in hand_chains]]:
        #     pose_stamped = PoseStamped()
        #     pose_stamped.header.frame_id = 'world'
        #     pose_stamped.pose.position.x = float(pos[0])
        #     pose_stamped.pose.position.y = float(pos[1])
        #     pose_stamped.pose.position.z = float(pos[2])
        #     pose_stamped.pose.orientation.w = 1.0  # Dummy

        #     # Transform to 'reference' frame
        #     try:
        #         tf = self.tf_buffer.lookup_transform('reference', 'world', rclpy.time.Time())
        #         pose_in_ref = do_transform_pose(pose_stamped, tf)
        #         q_out = self.solve_ik('index', pose_in_ref)
        #         if q_out:
        #             msg = JointState()
        #             msg.header.stamp = self.get_clock().now().to_msg()
        #             msg.name = [f"index_joint_{i+1}" for i in range(q_out.rows())]
        #             msg.position = [q_out[i] for i in range(q_out.rows())]
        #             self.joint_pub.publish(msg)
        #     except Exception as e:
        #         self.get_logger().warn(f"TF or IK failed: {e}")

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

    def publish_static_tf(self):
        # Step 1: Broadcast world → reference as before
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'reference'
        t.child_frame_id = 'manus'

        if not self.tuning_mode:
            t.transform.translation.x = self.reference_to_manus['x']
            t.transform.translation.y = self.reference_to_manus['y']
            t.transform.translation.z = self.reference_to_manus['z']
            t.transform.rotation.x = self.reference_to_manus['qx']
            t.transform.rotation.y = self.reference_to_manus['qy']
            t.transform.rotation.z = self.reference_to_manus['qz']
            t.transform.rotation.w = self.reference_to_manus['qw']
        else:
            t.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(t)
        self.get_logger().info("Published static transform world → reference")

def main(args=None):
    rclpy.init(args=args)
    node = HandAnimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
