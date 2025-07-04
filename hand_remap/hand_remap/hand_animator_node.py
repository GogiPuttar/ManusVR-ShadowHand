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
        self.get_logger().debug(f"Loaded {self.total_frames} frames")

        self.colors = [ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), # cyan
                       ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), # green
                       ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), # blue
                       ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), # red
                       ]

        self.mode = tracking_config.get('mode', 'IK')
        self.scaling_factor = tracking_config.get('scaling_factor', {})
        self.tuning_mode = tracking_config.get('tuning_mode', False)
        self.reference_to_manus = tracking_config['reference_to_manus']

        # Load URDF and parse KDL chains
        self.robot = URDF.from_xml_string(self.robot_description)
        self.log_urdf_tree(self.get_logger())

        success, self.tree = treeFromUrdfModel(self.robot)
        if not success:
            self.get_logger().error("Failed to parse URDF into KDL tree")
            return

        self.kdl_chains_middle = {}
        self.kdl_chains_distal = {}
        self.kdl_chains_tip = {}

        self.ik_solvers_middle = {}
        self.ik_solvers_distal = {}
        self.ik_solvers_tip = {}

        self.fk_solvers_middle = {}
        self.fk_solvers_distal = {}
        self.fk_solvers_tip = {}

        self.finger_links = {
            'thumb' : ['rh_thmiddle', 'rh_thdistal', 'rh_thtip'],
            'index' : ['rh_ffmiddle', 'rh_ffdistal', 'rh_fftip'],
            'middle': ['rh_mfmiddle', 'rh_mfdistal', 'rh_mftip'],
            'ring'  : ['rh_rfmiddle', 'rh_rfdistal', 'rh_rftip'],
            'little': ['rh_lfmiddle', 'rh_lfdistal', 'rh_lftip'],
        }
        self.last_q = {finger: None for finger in self.finger_links}

        for finger, segments in self.finger_links.items():

            # Chains
            middle_chain = self.tree.getChain('reference', segments[0])
            distal_chain = self.tree.getChain('reference', segments[1])
            tip_chain    = self.tree.getChain('reference', segments[2])

            self.kdl_chains_middle[finger] = middle_chain
            self.kdl_chains_distal[finger] = distal_chain
            self.kdl_chains_tip[finger]    = tip_chain

            # FK Solvers
            self.fk_solvers_middle[finger] = PyKDL.ChainFkSolverPos_recursive(middle_chain)
            self.fk_solvers_distal[finger] = PyKDL.ChainFkSolverPos_recursive(distal_chain)
            self.fk_solvers_tip[finger]    = PyKDL.ChainFkSolverPos_recursive(tip_chain)

            # IK Solvers
            self.ik_solvers_middle[finger] = PyKDL.ChainIkSolverPos_LMA(middle_chain)
            self.ik_solvers_distal[finger] = PyKDL.ChainIkSolverPos_LMA(distal_chain)
            self.ik_solvers_tip[finger]    = PyKDL.ChainIkSolverPos_LMA(tip_chain)

            self.get_logger().debug(f"KDL chain for {finger} middle has {middle_chain.getNrOfJoints()} joints.")
            self.get_logger().debug(f"KDL chain for {finger} distal has {distal_chain.getNrOfJoints()} joints.")
            self.get_logger().debug(f"KDL chain for {finger} tip has {tip_chain.getNrOfJoints()} joints.")
            self.get_logger().debug(f"")

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
                logger.debug(f"{indent}- {joint.child}  (joint: {joint.name}, type: {joint.type})")
                recurse(joint.child, indent + '  ')

        logger.debug(f"🤖 URDF Robot: {self.robot.name}")
        logger.debug(f"📦 URDF Link Tree:")
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

    def solve_position_only_ik(self, finger, target_pos, q_seed=None, tol=1e-4):

        chain = self.kdl_chains_tip[finger]
        fk_solver = self.fk_solvers_tip[finger]

        # chain = self.kdl_chains_middle[name]
        # fk_solver = self.fk_solvers_middle[name] # Remember to correctly format joint outputs when using this

        # chain = self.kdl_chains_distal[name]
        # fk_solver = self.fk_solvers_distal[name]

        n_joints = chain.getNrOfJoints()

        if q_seed is None:
            q_seed = np.zeros(n_joints)

        def fk_position_error(q):
            q_kdl = PyKDL.JntArray(n_joints)
            for i in range(n_joints):
                q_kdl[i] = q[i]

            error = 0.0
            fk_frame = PyKDL.Frame()

            fk_solver.JntToCart(q_kdl, fk_frame)
            # fk_solver.JntToCart(q_kdl, fk_frame, 0)
            pos = fk_frame.p

            p_vec = np.array([pos[0], pos[1], pos[2]])
            error += np.linalg.norm(p_vec - target_pos) ** 2

            return error

        result = minimize(fk_position_error, q_seed, method='BFGS')
        self.get_logger().debug(f"Target: {target_pos}")
        self.get_logger().debug(f"Optimized pos error: {fk_position_error(result.x):.4f}")

        final_error = fk_position_error(result.x)
        if result.success or final_error < tol:
            q_result = PyKDL.JntArray(n_joints)
            for i in range(n_joints):
                q_result[i] = result.x[i]
            return q_result
        else:
            self.get_logger().warn(f"IK failed (success={result.success}, error={final_error:.6f})")
            return None
        
    def solve_segment_position_only_ik(self, finger, target_positions, q_seed=None, weights=[1.0, 1.0, 1.0], tol=1e-4):

        n_joints = self.kdl_chains_tip[finger].getNrOfJoints()

        if q_seed is None:
            q_seed = np.zeros(n_joints)

        def fk_position_error(q):
            q_kdl = PyKDL.JntArray(n_joints)
            for i in range(n_joints):
                q_kdl[i] = q[i]

            error = 0.0
            fk_frame_middle = PyKDL.Frame()
            fk_frame_distal = PyKDL.Frame()
            fk_frame_tip = PyKDL.Frame()

            self.fk_solvers_middle[finger].JntToCart(q_kdl, fk_frame_middle)  
            self.fk_solvers_distal[finger].JntToCart(q_kdl, fk_frame_distal)  
            self.fk_solvers_tip[finger].JntToCart(q_kdl, fk_frame_tip)  

            # pos_middle = fk_frame_middle.p
            pos_distal = fk_frame_distal.p
            pos_tip    = fk_frame_tip.p

            # p_vec_middle = np.array([pos_middle[0], pos_middle[1], pos_middle[2]])
            p_vec_distal = np.array([pos_distal[0], pos_distal[1], pos_distal[2]])
            p_vec_tip    = np.array([pos_tip[0], pos_tip[1], pos_tip[2]])

            # error += weights[0] * (np.linalg.norm(p_vec_middle - target_positions[0]) ** 2)
            error += weights[1] * (np.linalg.norm(p_vec_distal - target_positions[1]) ** 2)
            error += weights[2] * (np.linalg.norm(p_vec_tip - target_positions[2]) ** 2)

            return error  # squared total error

        result = minimize(fk_position_error, q_seed, method='BFGS')
        self.get_logger().debug(f"Target: {target_positions}")
        self.get_logger().debug(f"Optimized pos error: {fk_position_error(result.x):.4f}")

        final_error = fk_position_error(result.x)
        if result.success or final_error < tol:
            q_result = PyKDL.JntArray(n_joints)
            for i in range(n_joints):
                q_result[i] = result.x[i]
            return q_result
        else:
            self.get_logger().warn(f"IK failed (success={result.success}, error={final_error:.6f})")
            return None

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
                pos[0] *= self.scaling_factor['x']
                pos[1] *= self.scaling_factor['y']
                pos[2] *= self.scaling_factor['z']

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

        joint_prefixes = {
            'thumb' : 'rh_TH',
            'index' : 'rh_FF',
            'middle' : 'rh_MF',
            'ring' : 'rh_RF',
            'little' : 'rh_LF',
        }
        finger_list = list(joint_prefixes)

        name_list = []
        position_list = []

        # for i in [1]:
        for i in range(5):

            # target = global_positions[4*i + 3]
            # target[0] *= self.scaling_factor['x']
            # target[1] *= self.scaling_factor['y']
            # target[2] *= self.scaling_factor['z']

            # target = np.array([0.04, -0.05, 0.1])
            # target = np.array([0.04, -0.02, 0.1])

            if not self.tuning_mode:
                target = [
                    global_positions[4 * i + 1] * self.scaling_factor['x'],  # middle
                    global_positions[4 * i + 2] * self.scaling_factor['y'],  # distal
                    global_positions[4 * i + 3] * self.scaling_factor['z']   # fingertip
                ]
            else:
                target = [
                    global_positions[4 * i + 1],  # middle
                    global_positions[4 * i + 2],  # distal
                    global_positions[4 * i + 3]   # fingertip
                ]

            # target = [
            #     np.array([0.04, -0.05, 0.1]),
            #     np.array([0.04, -0.04, 0.11]),
            #     np.array([0.04, -0.05, 0.1]),
            # ] 

            q_seed = self.last_q[finger_list[i]]
            # q_out = self.solve_position_only_ik(
            #     finger_list[i],
            #     target,
            #     q_seed=q_seed
            # )
            q_out = self.solve_segment_position_only_ik(
                finger_list[i],
                target,
                q_seed=q_seed,
                weights=[1.0, 1.0, 1.0]
            )

            if q_out is not None:
                self.last_q[finger_list[i]] = np.array([q_out[j] for j in range(q_out.rows())])

            name_list += [f"{joint_prefixes[finger_list[i]]}J{idx}" for idx in range(q_out.rows(), 0, -1)]
            position_list += [q_out[idx] for idx in range(q_out.rows())]
        
        name_list.append('rh_WRJ1')
        name_list.append('rh_WRJ2')
        position_list.append(0.0)
        position_list.append(0.0)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = name_list
        msg.position = position_list
        self.joint_pub.publish(msg)

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
