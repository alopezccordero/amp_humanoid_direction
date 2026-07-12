"""Shared, heading-invariant AMP feature construction.

WHY THIS FILE EXISTS (the main bug in the original code):
The discriminator was fed `concat[qpos[2:], qvel]`, which contains the root
quaternion in the WORLD frame and the root linear velocity in the WORLD frame.
Your task randomizes the walking direction every episode, but the mocap clips
walk in one fixed world direction. So the discriminator learns to separate
real/fake by *global heading*, not by gait quality -> the style reward
actively punishes the policy whenever it follows a target direction that
differs from the dataset's heading. Task reward and style reward fight each
other and neither wins.

Fix: express all root quantities in a local "heading frame" (yaw removed),
exactly like the AMP paper. Both the policy transitions (amp_env.py) and the
mocap transitions (motion_lib.py) MUST use this same function.

Feature layout (45 dims -> transition pair is 90 = discriminator input_dim):
    root height                                   1
    root orientation, yaw removed (quat, w>=0)    4
    root linear velocity in heading frame         3
    root angular velocity (MuJoCo body frame)     3
    joint angles                                 17
    joint velocities                             17
"""

import numpy as np

AMP_OBS_DIM = 45
AMP_TRANSITION_DIM = 2 * AMP_OBS_DIM


def quat_mul(a, b):
    """Hamilton product, MuJoCo [w, x, y, z] convention."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def yaw_from_quat(q):
    """ZYX-convention yaw (heading) of a [w, x, y, z] quaternion."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_amp_obs(qpos, qvel):
    """Humanoid-v5 layout:
        qpos = [x, y, z, qw, qx, qy, qz, 17 joint angles]        (24,)
        qvel = [vx, vy, vz (world), wx, wy, wz (body), 17 vels]  (23,)
    Returns float32 (45,) heading- and position-invariant features.
    """
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    yaw = yaw_from_quat(quat)

    c, s = np.cos(yaw), np.sin(yaw)
    # Rotates a world-frame vector into the heading frame (inverse yaw).
    R_inv = np.array([[c, s, 0.0],
                      [-s, c, 0.0],
                      [0.0, 0.0, 1.0]])

    # Remove yaw from the root orientation: q_local = q_yaw^-1 (x) q
    half = -0.5 * yaw
    q_yaw_inv = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    quat_local = quat_mul(q_yaw_inv, quat)
    if quat_local[0] < 0.0:
        quat_local = -quat_local  # canonical sign (quats double-cover)

    lin_vel_local = R_inv @ np.asarray(qvel[0:3], dtype=np.float64)
    # MuJoCo free-joint angular velocity (qvel[3:6]) is already expressed in
    # the body-local frame, i.e. heading-invariant. Left untouched. Either
    # way, expert and policy use the identical convention/transform.
    ang_vel = qvel[3:6]

    return np.concatenate([
        qpos[2:3],       # root height
        quat_local,      # roll/pitch information only
        lin_vel_local,   # forward/lateral/vertical speed relative to facing
        ang_vel,
        qpos[7:],        # joint angles (already local)
        qvel[6:],        # joint velocities (already local)
    ]).astype(np.float32)
