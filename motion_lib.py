import pickle
import random
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from amp_obs import build_amp_obs


def slerp(q0, q1, alpha):
    """Spherical interpolation of [w,x,y,z] quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * alpha
    sin0 = np.sin(theta0)
    return (np.sin(theta0 - theta) / sin0) * q0 + (np.sin(theta) / sin0) * q1


class MotionLib:
    """Loads retargeted mocap and serves AMP transitions.

    FIXES vs original:
    1. AMP features come from the shared `build_amp_obs` (heading-invariant),
       identical to what the policy side produces in amp_env.py.
    2. Expert transitions are sampled at the *exact* environment control dt by
       interpolating between mocap frames (lerp + quaternion slerp).
       Previously: frame_gap = max(1, round(dt*fps)) = max(1, round(0.015*30))
       = 1 frame = 33 ms, while policy transitions span 15 ms. Expert pairs
       showed ~2.2x larger per-transition displacement, so the discriminator
       could separate real/fake by displacement magnitude alone, regardless
       of gait quality.
    """

    def __init__(
        self,
        motion_dir="retargeted_pkl",
        filter_bad_contacts=True,
        max_lowest_foot_min=0.12,
        transition_dt=None,
    ):
        self.motion_dir = Path(motion_dir)
        self.transition_dt = transition_dt
        self.motions = []

        files = sorted(self.motion_dir.rglob("*.pkl"))
        if len(files) == 0:
            raise FileNotFoundError(f"No .pkl files found in {motion_dir}")

        if filter_bad_contacts:
            env = gym.make("Humanoid-v5")
            model = env.unwrapped.model
            data = env.unwrapped.data
            left_id = model.body("left_foot").id
            right_id = model.body("right_foot").id
        else:
            env = model = data = left_id = right_id = None

        loaded = 0
        skipped = 0

        for file in files:
            with open(file, "rb") as f:
                motion = pickle.load(f)

            if "qpos" not in motion or "qvel" not in motion:
                print("Skipping invalid file:", file)
                skipped += 1
                continue

            qpos = np.asarray(motion["qpos"], dtype=np.float64)
            qvel = np.asarray(motion["qvel"], dtype=np.float64)
            fps = float(motion.get("fps", 30.0))
            duration = (len(qpos) - 1) / fps

            min_duration = self.transition_dt if self.transition_dt else (1.0 / fps)
            if duration <= min_duration:
                print("Skipping too-short file:", file)
                skipped += 1
                continue

            if filter_bad_contacts:
                lowest_foot_min = self.compute_lowest_foot_min(
                    qpos, model, data, left_id, right_id
                )
                if lowest_foot_min >= max_lowest_foot_min:
                    print(
                        f"Skipping bad-contact motion: {file} "
                        f"lowest_foot_min={lowest_foot_min:.3f}"
                    )
                    skipped += 1
                    continue

            self.motions.append({
                "file": str(file),
                "fps": fps,
                "qpos": qpos,
                "qvel": qvel,
                "length": len(qpos),
                "duration": duration,
            })
            loaded += 1

        if env is not None:
            env.close()

        if len(self.motions) == 0:
            raise RuntimeError("No valid motions loaded after contact filtering.")

        print(f"Loaded {loaded} motions, skipped {skipped}")
        if self.transition_dt is not None:
            print(f"Expert transitions interpolated at dt = {self.transition_dt:.4f} s")

    def compute_lowest_foot_min(self, qpos_seq, model, data, left_id, right_id):
        lowest = float("inf")
        for qpos in qpos_seq:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            lowest = min(lowest, data.xpos[left_id][2], data.xpos[right_id][2])
        return lowest

    # ------------------------------------------------------------------ #

    def _interp_state(self, motion, t):
        """State at continuous time t (seconds) via lerp + quat slerp."""
        f = t * motion["fps"]
        i0 = min(int(np.floor(f)), motion["length"] - 2)
        alpha = float(f - i0)

        qpos0, qpos1 = motion["qpos"][i0], motion["qpos"][i0 + 1]
        qvel0, qvel1 = motion["qvel"][i0], motion["qvel"][i0 + 1]

        qpos = (1.0 - alpha) * qpos0 + alpha * qpos1
        qpos[3:7] = slerp(qpos0[3:7], qpos1[3:7], alpha)
        qvel = (1.0 - alpha) * qvel0 + alpha * qvel1
        return qpos, qvel

    def sample_amp_transition(self):
        motion = random.choice(self.motions)

        if self.transition_dt is None:
            i = random.randint(0, motion["length"] - 2)
            s0 = (motion["qpos"][i], motion["qvel"][i])
            s1 = (motion["qpos"][i + 1], motion["qvel"][i + 1])
        else:
            t0 = random.uniform(0.0, motion["duration"] - self.transition_dt)
            s0 = self._interp_state(motion, t0)
            s1 = self._interp_state(motion, t0 + self.transition_dt)

        return np.concatenate([
            build_amp_obs(*s0),
            build_amp_obs(*s1),
        ]).astype(np.float32)

    def sample_reference_state(self):
        motion = random.choice(self.motions)
        i = random.randint(0, motion["length"] - 1)
        return motion["qpos"][i].copy(), motion["qvel"][i].copy()

    def compute_amp_stats(self, num_samples=10000):
        samples = np.array(
            [self.sample_amp_transition() for _ in range(num_samples)],
            dtype=np.float32,
        )
        mean = samples.mean(axis=0).astype(np.float32)
        std = samples.std(axis=0).astype(np.float32) + 1e-6
        return mean, std
