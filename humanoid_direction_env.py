import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.utils import EzPickle
from gymnasium import spaces


class HumanoidDirectionEnv(HumanoidEnv, EzPickle):
    """Humanoid-v5 with a random target heading each episode.

    FIX vs original: the task reward is now BOUNDED in [0, 1], AMP-paper style.
    Old reward: 5.0 * dot(v, dir) + 0.5 * healthy - ctrl_cost
      - Unbounded in speed -> PPO maximizes it by sprinting/lunging, which is
        exactly the behavior the motion prior punishes, so the two rewards
        fight each other.
      - Its scale (~2.5 to 7+ per step) dwarfed the [0, 0.2] style reward, so
        the discriminator was effectively ignored.
    New reward: full credit once velocity along the target reaches
    `target_speed` (default 1.4 m/s, a normal walking pace present in mocap),
    no extra credit for going faster. Falling is handled by termination and by
    the style reward (mocap contains no falling), so the constant alive bonus
    is gone.
    """

    def __init__(self, direction=(1.0, 0.0), target_speed=1.4, **kwargs):
        self.target_dir = np.array(direction, dtype=np.float32)
        self.target_dir /= np.linalg.norm(self.target_dir)
        self.target_speed = float(target_speed)

        EzPickle.__init__(self, direction, target_speed, **kwargs)
        super().__init__(**kwargs)

        old_space = self.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(old_space.shape[0] + 2,),
            dtype=np.float64,
        )

    def reset_model(self):
        obs = super().reset_model()
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.target_dir = np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        return self._get_obs_with_direction(obs)

    def step(self, action):
        xy_before = self.data.xpos[self.model.body("torso").id][:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.xpos[self.model.body("torso").id][:2].copy()

        xy_velocity = (xy_after - xy_before) / self.dt
        v_proj = float(np.dot(xy_velocity, self.target_dir))

        # Only penalize the *deficit* below target speed; exceeding it earns
        # nothing extra. Reward is in [0, 1].
        speed_deficit = max(0.0, self.target_speed - v_proj)
        direction_reward = float(np.exp(-2.0 * speed_deficit ** 2))

        reward = direction_reward
        terminated = not self.is_healthy
        obs = self._get_obs_with_direction(self._get_obs())

        info = {
            "xy_velocity": xy_velocity,
            "target_dir": self.target_dir,
            "v_proj": v_proj,
            "direction_reward": direction_reward,
        }
        return obs, reward, terminated, False, info

    def _get_obs_with_direction(self, humanoid_obs):
        return np.concatenate([humanoid_obs, self.target_dir]).astype(np.float64)
