import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
import numpy as np
import sys


def quaternion_rotate(v, q):
    # q = [x, y, z, w]
    q = np.array([q[0], q[1], q[2], q[3]])
    v = np.array([v[0], v[1], v[2], 0.0])
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    v_rot = quaternion_multiply(quaternion_multiply(q, v), q_conj)
    return v_rot[0:3]


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])


def animate_skeleton(csv_path):

    df = pd.read_csv(csv_path)

   # Extract all position columns
    pos_cols = [c for c in df.columns if re.match(r'p\d+_[xyz]', c)]
    node_positions = df[pos_cols].values.reshape(
        len(df), -1, 3)  # shape: (frames, joints, 3)

    # get orientations
    orient_cols = [c for c in df.columns if re.match(r'o\d+_[xyzw]', c)]
    node_orientations = df[orient_cols].values.reshape(
        len(df), -1, 4)  # shape: (frames, joints, 4)

    # only take every 2nd frame for animation to speed it up
    node_positions = node_positions[::2]
    node_orientations = node_orientations[::2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)  # Adjust view angle for better visibility
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    # Define repeating colors
    colors = ['cyan', 'green', 'blue', 'red']
    num_joints = node_positions.shape[1]-1
    color_cycle = [colors[i % len(colors)] for i in range(num_joints)]

    initial_pos = np.zeros((num_joints, 3))
    scat = ax.scatter(initial_pos[:, 0], initial_pos[:, 1],
                      initial_pos[:, 2], c=color_cycle, s=50)

    def init():
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
        ax.set_title("Skeleton Animation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return scat,

    def update(frame):
        hand_chains = [[1, 2, 3, 4], [5, 6, 7, 8], [
            9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        global_positions = []
        for chain in hand_chains:
            # always start with root node position (always 0,0,0)
            parent_position = np.array([0, 0, 0])
            parent_orientation = np.array([0, 0, 0, 1])
            marker_id = -1
            for i in range(len(chain)):
                # compute global position based on quaternion
                orientation = quaternion_multiply(parent_orientation, np.array(
                    [node_orientations[frame][chain[i]][0], node_orientations[frame][chain[i]][1], node_orientations[frame][chain[i]][2], node_orientations[frame][chain[i]][3]]))
                position = parent_position + \
                    quaternion_rotate(np.array([node_positions[frame][chain[i]][0],
                                                node_positions[frame][chain[i]][1], node_positions[frame][chain[i]][2]]), parent_orientation)
                parent_position = position
                parent_orientation = orientation
                global_positions.append(position)
        # Update scatter plot with new positions
        pos = np.array(global_positions)
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        return scat,

    ani = FuncAnimation(fig, update, frames=len(node_positions),
                        init_func=init, blit=False, interval=100)
    plt.show()
    return ani


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python animate_skeleton.py <path_to_csv_file>")
        sys.exit(1)
    else:
        csv_path = sys.argv[1]
        ani = animate_skeleton(csv_path)
