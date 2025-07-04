# ManusVR-ShadowHand

# Overview
This repo describes a ROS 2 implementation of remapping a Manus VR Glove's position and orientation measurements onto a ShadowRobot Dexterous Hands through:

1. Reconstruction of Hand Chains using relative position measurements and relative orientation measurements.
2. Remapping of tracked position space to robot target position space, using a linear transform.
3. Segmentwise Inverse Kinematics to get robot joint positions.

This repo does not consider self collisions and actuator control characteristics (see [limitations](#limitations))

https://github.com/user-attachments/assets/0624f13a-9e5d-4638-8620-6eaa8df2d4e7

## Usage
```
ros2 launch hand_remap hand_animator.launch.py csv_filename:=skeleton_log_20250624_203229.csv
```

<br>

# Mathematical Foundation

## Remapping

Since the Manus VR glove and the Shadow Hand have a similar tree structure with different geometries, it is possible and necessary to remap the space of tracked positions ($X \in ℝ^3$) to the cartesian workspace of the Shadow Hands ($Y \in ℝ^3$) with a suitable transform $T$.

$Y = T(X)$

Since it is important to preserve addition (Eg. offsets to any positions) and scaling (Eg. lengthening of any relative positions), I propose using using an invertible $3⨯3$ linear transformation $S$, such that:

$Y = SX$

This is a general transformer that preserves collinearity, parallelism and origin, and represents an $8$-dimensional manifold in $9$-dimensional space.
Since tuning 9 different values is tough, I've decided to use a simple scaling matrix $S = diag(s_x, s_y, s_z)$ where $s_x$, $s_y$, and $s_z$ are the respective scaling factors for $x$, $y$, and $z$ axes.

https://github.com/user-attachments/assets/04cc70e6-ae19-4ee7-9ca7-9ae67832159a

<br>

## Inverse Kinematics

A segment-wise weighted IK solver has been implemented on each finger for generating appropriate joint values, given the scaled Manus VR glove's tracking trajectories.

Since each finger of the ShadowHand is 4-5 DoF and the fact that fingertip pose needs to be inferred using position values from the VR glove, it is non-trivial to have 6-DoF $SE(3)$ pose tracking for such a case.
Therefore, `PyKDL`'s builtin IK Solver can't be used directly as it relies on 6-DoF pose commands.
I've implemented my own IK solver (built on `PyKDL` primitives) as follows:

### Tip Position-only IK:

Position only IK for the fingertips is defined as the unconstrained optimization problem:

For each finger,

$\min_{q} Z_{tip}(q)$

where,

$Z_{tip}(q) = (x_{d,tip} - FK_{tip}(q))^2$ is the cost function,

$x_{d,tip} \in ℝ^3$ is the desired fingertip position,

$FK_{tip}(q)$ is the Cartesian Forward Kinematics function for the respective finger, and

$q \in ℝ^n$ is the vector of joint angles ($n$ is the DoF for the respective finger).

As seen below, position only IK for the tip suffers from 1-2 dimensional redundancies, and is therefore unable to accurately represent the tracked positions fully:

https://github.com/user-attachments/assets/ac022014-75cc-446a-8c52-2db2a683cc90

### Segment-wise Position-only IK:

The segment positions provided by the Manus glove can be correlated with existing links on the fingers of the ShadowHand, namely `middle`, `distal`, and `tip` (knuckles are ignored).

Therefore, the position only segmentwise IK solver can be defined as the unconstrained optimization problem:

For each finger,

$\min_{q} w_1 Z_{middle}(q) + w_2 Z_{distal}(q) + w_3 Z_{tip}(q)$

where,

$Z_{middle}(q) = (x_{d,middle} - FK_{middle}(q))^2$,

$Z_{distal}(q) = (x_{d,distal} - FK_{distal}(q))^2$,

$Z_{tip}(q) = (x_{d,tip} - FK_{tip}(q))^2$,

are the individual cost functions, 

$w_1, w_2, w_3 \in ℝ$, 

are the respective weights, and $x_{d,name}$, $FK_{name}(q)$ follow the same logic.

This yields more accurate solutions, which can be tuned to perfection by correctly correlating glove sensor positions to robot links:

https://github.com/user-attachments/assets/43832bee-6af8-4fbd-b4e0-add73fe6bdcb

**NOTES:**
- `scipy.minimize(method=’BFGS’)` has been used for optimization, since it works. No benchmarking has been done but the node seems to work well without any issues at 10Hz.
- The previous solution `q_seed` is given as the starting solution to the optimizer.
- A tolerance value is added to account for target positions outside the workspace, which gives us clean reasonable approximations instead of failures.
- Since the search space is unconstrained, the optimizer has no issues with values outside the workspace due to joint limits, again giving us a clean solution limited by the renderer's (`robot_state_publisher`) joint limits.
- Since this is position-only, sometimes you see the fingers twisting around which is undesriable. The optimizer can be modified to penalize joint twisting.

<br>

# Implementation

## `scripts/aniamte_skeleton.py`

Simple script for Manus VR glove position reconstruction and animation using quaternion math.
`data` and `animate_skeleton.py` have been provided by [Futurhand Robotics](https://futurhandrobotics.com/).
```
python3 scripts/animate_skeleton.py data/skeleton_log_20250624_203941.csv
```

https://github.com/user-attachments/assets/fdc4988d-ea6c-4606-aa97-6befc9a6492a

## `hand_remap` package

ROS 2 package that simulates remapping Manus VR measurements onto a ShadowRobot Hand in real time.

```
ros2 launch hand_remap hand_animator.launch.py csv_filename:=skeleton_log_20250624_203229.csv
```

## `sr_description` pacakge

ROS 2 package that stores the relevant `urdf` and `xacro` files for simualting a ShadowRobot Hand.

<br>

# Setup

## REQUIREMENTS:
- ROS 2 Humble (This code was tested on Ubuntu 22.04+)
- ROS 2 TF2 Transformations 
  ```
  sudo apt install ros-humble-tf*
  ```
- Kinematics and Dynamics Library (KDL) Parser (`kdl_parser_py` package). See intructions for installation, and adaptation to ROS 2 [here](https://github.com/GogiPuttar/Bi-ManualManipulation/tree/main?tab=readme-ov-file#requirements). 
  
- The `sr_description` ROS 2 package has been adapted from the [`sr_common` ROS 1 repo](https://github.com/shadow-robot/sr_common), via:
  ```
  mv sr_common/sr_description/robots/ ManusVR-ShadowHand/sr_description/robots
  mv sr_common/sr_description/meshes/ ManusVR-ShadowHand/sr_description/meshes
  mv sr_common/sr_description/hand/ ManusVR-ShadowHand/sr_description/hand
  mv sr_common/sr_description/other/ ManusVR-ShadowHand/sr_description/other
  rm -rf sr_common/
  ```

<br>

# Limitations
- Does not account for self collisions, although good remapping and IK can minimize the possibility of self collisions. 
Can be baked into the IK optimizer, possibly as a Gaussian Mixture Model.
- Only accounts for joint limits, but does not account for actuator gains, damping or any other real world considerations.

<br>

# Use of AI Tools
ChatGPT was used in this project for creating protoype scripts