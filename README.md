# ManusVR-ShadowHand

# Overview
Remapping Manus VR glove motion extraction and remapping to ShadowRobot Dexterous Hand

https://github.com/user-attachments/assets/0624f13a-9e5d-4638-8620-6eaa8df2d4e7

## Usage
```
ros2 launch hand_remap hand_animator.launch.py csv_filename:=skeleton_log_20250624_203229.csv
```

# Mathematical Foundation

## Remapping
- Assume a transform from Manus VR tracked positions to arbitrary hand's target positions. this must preserve linear combinations. also gives us collinearity, parallelism, fixed origin. all invertible 3x3 transforms. scaling ones used, don't have to tune 9 different values.
- quaternion formula for manus position chain reconstruction

https://github.com/user-attachments/assets/04cc70e6-ae19-4ee7-9ca7-9ae67832159a

## Inverse Kinematics
- Gives closest IK solutions for points outside the workspace. SCIPY BFGS. Benchmarking not done but performs well so far at 10Hz.
- Currently, does not care about the pose of the finger, that's why sometimes you see the finger twist around. Joint-wise weighting can be added.

### Tip-Only IK:
https://github.com/user-attachments/assets/ac022014-75cc-446a-8c52-2db2a683cc90

### Segment-wise IK:
https://github.com/user-attachments/assets/43832bee-6af8-4fbd-b4e0-add73fe6bdcb

# Implementation

## `scripts/aniamte_skeleton.py`

`data` and `animate_skeleton.py` have been provided by [Futurhand Robotics](https://futurhandrobotics.com/).
```
python3 scripts/animate_skeleton.py data/skeleton_log_20250624_203941.csv
```

https://github.com/user-attachments/assets/fdc4988d-ea6c-4606-aa97-6befc9a6492a

## `hand_remap` package

```
ros2 launch hand_remap hand_animator.launch.py csv_filename:=skeleton_log_20250624_203229.csv
```

## `sr_description` pacakge


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

# Limitations
- Does not account for self collisions, although good remapping and IK can minimize the possibility of self collisions. 
Can be baked into the IK optimizer, possibly as a Gaussian Mixture Model.
- Only accounts for joint limits, but does not account for actuator gains, damping or any other real world considerations.

# Use of AI Tools
ChatGPT was used in this project for creating protoype scripts