import numpy as np
import torch
import gymnasium as gym

import register_env  # noqa: F401  (registers HumanoidDirection-v0)
from amp_obs import build_amp_obs


class AMPHumanoidEnv(gym.Wrapper):
    """Wraps HumanoidDirection-v0 with an AMP style reward.

    FIXES vs original:
    - AMP features come from the shared, heading-invariant `build_amp_obs`
      (was: raw world-frame qpos[2:]+qvel, letting the discriminator cheat on
      global heading).
    - Reward mixing is paper-style: total = w_task * r_task + w_style * r_style
      with both rewards in [0, 1] and default weights 0.5 / 0.5
      (was: r_task in ~[0, 7] plus 0.2 * near-constant style reward).
    - `load_disc_state` lets the training callback push fresh discriminator
      weights into each env. This is REQUIRED under SubprocVecEnv, where every
      worker process holds its own copy of the discriminator; without syncing,
      the style reward would be computed with the frozen initial weights
      forever.
    """

    def __init__(
        self,
        discriminator,
        motion_lib=None,
        env_id="HumanoidDirection-v0",
        render_mode=None,
        task_weight=0.5,
        amp_weight=0.5,
        device="cpu",
        amp_mean=None,
        amp_std=None,
        reference_state_init_prob=0.5,
    ):
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.disc = discriminator
        self.disc.eval()
        self.motion_lib = motion_lib
        self.task_weight = task_weight
        self.amp_weight = amp_weight
        self.device = device
        self.prev_amp_obs = None
        self.fake_amp_transitions = []
        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.reference_state_init_prob = reference_state_init_prob

    # ------------------------------------------------------------------ #

    def get_amp_obs(self):
        data = self.env.unwrapped.data
        return build_amp_obs(data.qpos.copy(), data.qvel.copy())

    def load_disc_state(self, state_dict):
        """Called via VecEnv.env_method by the AMP callback after each
        discriminator update."""
        self.disc.load_state_dict(state_dict)
        self.disc.eval()
        return True

    def _current_policy_obs(self):
        unwrapped = self.env.unwrapped
        humanoid_obs = unwrapped._get_obs()
        if hasattr(unwrapped, "_get_obs_with_direction"):
            return unwrapped._get_obs_with_direction(humanoid_obs)
        return humanoid_obs

    def maybe_reference_state_init(self):
        if self.motion_lib is None or self.reference_state_init_prob <= 0.0:
            return False

        rng = self.env.unwrapped.np_random
        if rng.random() >= self.reference_state_init_prob:
            return False

        qpos, qvel = self.motion_lib.sample_reference_state()
        self.env.unwrapped.set_state(qpos, qvel)
        return True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        used_reference_state = self.maybe_reference_state_init()
        if used_reference_state:
            obs = self._current_policy_obs()

        self.prev_amp_obs = self.get_amp_obs()
        info["reference_state_init"] = used_reference_state
        return obs, info

    def step(self, action):
        obs, task_reward, terminated, truncated, info = self.env.step(action)

        current_amp_obs = self.get_amp_obs()
        amp_transition = np.concatenate(
            [self.prev_amp_obs, current_amp_obs]
        ).astype(np.float32)
        self.fake_amp_transitions.append(amp_transition)

        x_np = amp_transition
        if self.amp_mean is not None and self.amp_std is not None:
            x_np = (x_np - self.amp_mean) / self.amp_std

        with torch.no_grad():
            x = torch.tensor(
                x_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            amp_reward = self.disc.amp_reward(x).item()

        total_reward = (
            self.task_weight * task_reward + self.amp_weight * amp_reward
        )

        info["task_reward"] = task_reward
        info["amp_reward"] = amp_reward
        info["total_reward"] = total_reward

        self.prev_amp_obs = current_amp_obs
        return obs, total_reward, terminated, truncated, info

    def pop_fake_transitions(self):
        transitions = self.fake_amp_transitions
        self.fake_amp_transitions = []
        return transitions
