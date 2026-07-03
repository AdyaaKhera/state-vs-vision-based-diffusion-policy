"""
this module implements the kinematics for a 3 dof robot arm with revolute joints
this has all the functions for forward kinematics, inverse kinematics, jacobian, and angle wrapping
forward kinematics is when the joint angles are known and the end effector position is calculated
inverse kinematics is when the end effector position is known and the joint angles are calculated
jacobian is the matrix that relates the joint velocities to the end effector velocities
angle wrapping function is used to keep the joint angles within the range of -pi to pi
"""

from __future__ import annotations
import numpy as np
IDENTITY2 = np.eye(2, dtype=float)

#these arrays are the default link lengths and base position of the robot arm
DEFAULT_LINK_LENGTHS = np.array([0.26, 0.22, 0.16], dtype=float)
DEFAULT_BASE = np.array([0.02, 0.16], dtype=float)

"""this function wraps the angles to be within the range of -pi to pi by using the arctan2 function"""

def wrap_angles(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))

"""this function calculates the difference between two angles and wraps the result to be 
within the range of -pi to pi to calculate the shortest angular distance"""

def angle_difference(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return wrap_angles(np.asarray(target, dtype=float) - np.asarray(current, dtype=float))

"""this function calculates the forward kinematics of the robot arm
it takes in the joint angles, link lengths, and base position and returns the positions of each joint and the end effector in a 2D space
the positions are calculated by iteratively adding the link lengths and angles to the base position using trigonometric functions"""

def forward_kinematics(
    joints: np.ndarray,
    link_lengths: np.ndarray = DEFAULT_LINK_LENGTHS,
    base: np.ndarray = DEFAULT_BASE,) -> np.ndarray:
    
    joints = np.asarray(joints, dtype=float)
    link_lengths = np.asarray(link_lengths, dtype=float)
    base = np.asarray(base, dtype=float)

    points = np.zeros((len(link_lengths) + 1, 2), dtype=float) #setting up the array to hold the positions of each joint and the end effector
    points[0] = base #first point is the base position of the robot arm
    cumulative_angles = np.cumsum(joints) #cumulative sum of the joint angles to get the total angle at each joint as each joint moves relative to the previous one

    for idx, (angle, length) in enumerate(zip(cumulative_angles, link_lengths, strict=True), start=1):
        points[idx, 0] = points[idx - 1, 0] + length * np.cos(angle)
        points[idx, 1] = points[idx - 1, 1] + length * np.sin(angle)
    return points #points now is an array of shape (4, 2) where each row is the position of a joint or the end effector in 2D space

"""this function calculates the position of the end effector of the robot arm"""

def end_effector_position(
    joints: np.ndarray,
    link_lengths: np.ndarray = DEFAULT_LINK_LENGTHS,
    base: np.ndarray = DEFAULT_BASE,) -> np.ndarray:

    joints = np.asarray(joints, dtype=float)
    link_lengths = np.asarray(link_lengths, dtype=float)
    base = np.asarray(base, dtype=float)

    angles = np.cumsum(joints)

    x = base[0]
    y = base[1]

    for angle, length in zip(angles, link_lengths, strict=True):
        x += length * np.cos(angle) #Lcosine(theta) gives the x component of the link length
        y += length * np.sin(angle) #Lsin(theta) gives the y component of the link length

    return np.array([x, y], dtype=float)

"""this function calculates the jacobian matrix of the robot arm
the jacobian matrix relates the joint velocities to the end effector velocities
it takes in the joint angles and link lengths and returns a 2xN matrix where N is the number of joints
the first row of the matrix represents the x velocities and 
the second row represents the y velocities of the end effector with respect to the joint angles"""

def jacobian(
    joints: np.ndarray,
    link_lengths: np.ndarray = DEFAULT_LINK_LENGTHS,) -> np.ndarray:

    joints = np.asarray(joints, dtype=float)
    link_lengths = np.asarray(link_lengths, dtype=float)

    angles = np.cumsum(joints)

    sine = np.sin(angles)
    cosine = np.cos(angles)

    jac = np.zeros((2, len(link_lengths)), dtype=float) #since there's 2 rows for x and y velocities and N columns for each joint

    sine_sums = np.cumsum((link_lengths * sine)[::-1])[::-1]
    cosine_sums = np.cumsum((link_lengths * cosine)[::-1])[::-1]

    jac[0] = -sine_sums
    jac[1] = cosine_sums

    return jac

"""this function calculates the inverse kinematics of the robot arm
it takes in the target position of the end effector, initial joint angles, link lengths, 
base position, maximum number of iterations, damping factor, and step scale
it returns the joint angles that will achieve the target position of the end effector"""

def inverse_kinematics(
    target: np.ndarray,
    initial_joints: np.ndarray,
    link_lengths: np.ndarray = DEFAULT_LINK_LENGTHS,
    base: np.ndarray = DEFAULT_BASE,
    max_iterations: int = 64,
    damping: float = 0.08,
    step_scale: float = 0.85,
) -> np.ndarray:
    """
    Calculates the inverse kinematics of the robot arm using the
    damped least squares method.
    """

    target = np.asarray(target, dtype=float)
    joints = np.asarray(initial_joints, dtype=float).copy()
    link_lengths = np.asarray(link_lengths, dtype=float)
    base = np.asarray(base, dtype=float)

    # Maximum distance the robot can reach
    max_reach = np.sum(link_lengths)

    # Distance from the base to the target
    direction = target - base
    distance = np.linalg.norm(direction)

    # Clamp unreachable targets to the edge of the workspace
    if distance > max_reach and distance > 0:
        direction /= distance
        target = base + direction * max_reach

    for _ in range(max_iterations):

        end_effector = end_effector_position(
            joints,
            link_lengths=link_lengths,
            base=base,
        )

        error = target - end_effector

        # Avoid the square root (slightly faster)
        if error @ error < 1e-8:
            break

        jac = jacobian(joints, link_lengths=link_lengths)

        system = (
            jac @ jac.T
            + (damping ** 2) * IDENTITY2
        )

        delta = jac.T @ np.linalg.solve(system, error)

        joints = wrap_angles(
            joints + step_scale * delta
        )

    return joints